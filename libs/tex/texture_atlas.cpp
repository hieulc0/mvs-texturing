/*
 * Copyright (C) 2015, Nils Moehrle
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include <cstdio>
#include <iostream>
#include <map>

#include <util/file_system.h>
#include <util/timer.h>
#include <mve/image_tools.h>
#include <mve/image_io.h>

#include "texture_atlas.h"

#ifdef MVSTEX_GPU
#include "cuda/edge_padding_gpu.h"
#endif

double atlas_insert_binfit_time_sec = 0.0;
double atlas_insert_copy_time_sec = 0.0;


TextureAtlas::TextureAtlas(unsigned int size, mve::ImageType type, bool grayscale) :
    size(size), padding(size >> 7), finalized(false), grayscale(grayscale) {

    bin = RectangularBin::create(size, size);

    if (type == mve::IMAGE_TYPE_FLOAT){
        image = mve::FloatImage::create(size, size, 3);
    }else if (type == mve::IMAGE_TYPE_UINT16){
        image = mve::RawImage::create(size, size, 3);
    }else{
        image = mve::ByteImage::create(size, size, 3);
    }
    validity_mask = mve::ByteImage::create(size, size, 1);
}

/**
  * Copies the src image into the dest image at the given position,
  * optionally adding a border.
  * @warning asserts that the given src image fits into the given dest image.
  */
template <typename T>
void copy_into(typename mve::Image<T>::ConstPtr src, int x, int y,
    typename mve::Image<T>::Ptr dest, int border = 0) {

    assert(x >= 0 && x + src->width() + 2 * border <= dest->width());
    assert(y >= 0 && y + src->height() + 2 * border <= dest->height());

    for (int i = 0; i < src->width() + 2 * border; ++i) {
        for(int j = 0; j < src->height() + 2 * border; j++) {
            int sx = i - border;
            int sy = j - border;

            if (sx < 0 || sx >= src->width() || sy < 0 || sy >= src->height())
                continue;

            for (int c = 0; c < src->channels(); ++c) {
                dest->at(x + i, y + j, c) = src->at(sx, sy, c);
            }
        }
    }
}

mve::RawImage::Ptr
float_to_raw_image (mve::FloatImage::ConstPtr image, float vmin, float vmax)
{
    if (image == nullptr)
        throw std::invalid_argument("Null image given");

    mve::RawImage::Ptr img = mve::RawImage::create();
    img->allocate(image->width(), image->height(), image->channels());
    for (int i = 0; i < image->get_value_amount(); ++i)
    {
        float value = std::min(vmax, std::max(vmin, image->at(i)));
        value = 65535.0f * (value - vmin) / (vmax - vmin);
        img->at(i) = static_cast<uint16_t>(value + 0.5f);
    }
    return img;
}

typedef std::vector<std::pair<int, int> > PixelVector;

bool
TextureAtlas::insert(TexturePatch::ConstPtr texture_patch) {
    if (finalized) {
        throw util::Exception("No insertion possible, TextureAtlas already finalized");
    }

    assert(bin != NULL);
    assert(validity_mask != NULL);

    int const width = texture_patch->get_width() + 2 * padding;
    int const height = texture_patch->get_height() + 2 * padding;
    Rect<int> rect(0, 0, width, height);
    util::WallTimer binfit_timer;
    bool const placed = bin->insert(&rect);
    atlas_insert_binfit_time_sec += binfit_timer.get_elapsed_sec();
    if (!placed) return false;

    util::WallTimer copy_timer;

    /* Update texture atlas and its validity mask. */


    if (image->get_type() == mve::IMAGE_TYPE_FLOAT){
        copy_into<float>(texture_patch->get_image(), rect.min_x, rect.min_y, std::dynamic_pointer_cast<mve::FloatImage>(image), padding);
    }else if (image->get_type() == mve::IMAGE_TYPE_UINT16){
        mve::RawImage::Ptr patch_image = float_to_raw_image(
                texture_patch->get_image(), 0.0f, 1.0f);
        copy_into<uint16_t>(patch_image, rect.min_x, rect.min_y, std::dynamic_pointer_cast<mve::RawImage>(image), padding);
    }else{
        mve::ByteImage::Ptr patch_image = mve::image::float_to_byte_image(
                texture_patch->get_image(), 0.0f, 1.0f);
        copy_into<uint8_t>(patch_image, rect.min_x, rect.min_y, std::dynamic_pointer_cast<mve::ByteImage>(image), padding);
    }

    mve::ByteImage::ConstPtr patch_validity_mask = texture_patch->get_validity_mask();
    copy_into<uint8_t>(patch_validity_mask, rect.min_x, rect.min_y, validity_mask, padding);

    TexturePatch::Faces const & patch_faces = texture_patch->get_faces();
    TexturePatch::Texcoords const & patch_texcoords = texture_patch->get_texcoords();

    /* Calculate the offset of the texture patches' relative texture coordinates */
    math::Vec2f offset = math::Vec2f(rect.min_x + padding, rect.min_y + padding);

    faces.insert(faces.end(), patch_faces.begin(), patch_faces.end());

    /* Calculate the final textcoords of the faces. */
    for (std::size_t i = 0; i < patch_faces.size(); ++i) {
        for (int j = 0; j < 3; ++j) {
            math::Vec2f rel_texcoord(patch_texcoords[i * 3 + j]);
            math::Vec2f texcoord = rel_texcoord + offset;

            texcoord[0] = texcoord[0] / this->size;
            texcoord[1] = texcoord[1] / this->size;
            texcoords.push_back(texcoord);
        }
    }
    atlas_insert_copy_time_sec += copy_timer.get_elapsed_sec();
    return true;
}



template <typename T>
void
TextureAtlas::apply_edge_padding(bool use_gpu) {
    (void) use_gpu; /* GPU path only exists for the uint8_t specialization below. */
    this->apply_edge_padding_cpu<T>();
}

template <typename T>
void
TextureAtlas::apply_edge_padding_cpu(void) {
    assert(image != NULL);
    assert(validity_mask != NULL);

    const int width = image->width();
    const int height = image->height();
    typename mve::Image<T>::Ptr img = std::dynamic_pointer_cast<mve::Image<T>>(image);

    math::Matrix<float, 3, 3> gauss;
    gauss[0] = 1.0f; gauss[1] = 2.0f; gauss[2] = 1.0f;
    gauss[3] = 2.0f; gauss[4] = 4.0f; gauss[5] = 2.0f;
    gauss[6] = 1.0f; gauss[7] = 2.0f; gauss[8] = 1.0f;
    gauss /= 16.0f;

    /* Calculate the set of invalid pixels at the border of texture patches.
     * Rows are independent (each pixel is visited by exactly one thread),
     * so this scan is safe to parallelize; each thread accumulates locally
     * and merges once to avoid contending on a shared container. */
    PixelVector invalid_border_pixels;
    #pragma omp parallel
    {
        PixelVector local_border_pixels;
        #pragma omp for schedule(static) nowait
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                if (validity_mask->at(x, y, 0) == 255) continue;

                /* Check the direct neighbourhood of all invalid pixels. */
                bool is_border = false;
                for (int j = -1; j <= 1 && !is_border; ++j) {
                    for (int i = -1; i <= 1 && !is_border; ++i) {
                        int nx = x + i;
                        int ny = y + j;
                        /* If the invalid pixel has a valid neighbour: */
                        if (0 <= nx && nx < width &&
                            0 <= ny && ny < height &&
                            validity_mask->at(nx, ny, 0) == 255) {
                            is_border = true;
                        }
                    }
                }
                if (is_border) {
                    local_border_pixels.push_back(std::pair<int, int>(x, y));
                }
            }
        }
        #pragma omp critical
        invalid_border_pixels.insert(invalid_border_pixels.end(),
            local_border_pixels.begin(), local_border_pixels.end());
    }

    mve::ByteImage::Ptr new_validity_mask = validity_mask->duplicate();

    /* Flat frontier membership grid, replacing the previous std::set-based
     * frontier: this loop runs `padding` times over a border that can span
     * a large fraction of a multi-thousand-pixel atlas, so O(1) membership
     * tests matter more here than in most call sites. Only entries actually
     * touched by the current frontier are set/cleared each round, so this
     * stays proportional to frontier size, not image size. */
    std::vector<uint8_t> in_frontier(static_cast<std::size_t>(width) * height, 0);
    for (std::pair<int, int> const & p : invalid_border_pixels) {
        in_frontier[p.second * width + p.first] = 1;
    }

    /* Iteratively dilate border pixels until padding constants are reached. */
    for (unsigned int n = 0; n <= padding; ++n) {
        PixelVector new_valid_pixels;

        /* Each frontier pixel only writes its own image pixel and only
         * reads `new_validity_mask`, which this round doesn't mutate until
         * after this loop -- safe to parallelize without locks. */
        #pragma omp parallel
        {
            PixelVector local_valid_pixels;
            #pragma omp for schedule(dynamic) nowait
            for (std::size_t idx = 0; idx < invalid_border_pixels.size(); ++idx) {
                int x = invalid_border_pixels[idx].first;
                int y = invalid_border_pixels[idx].second;

                bool now_valid = false;
                /* Calculate new pixel value. */
                for (int c = 0; c < 3; ++c) {
                    float norm = 0.0f;
                    float value = 0.0f;
                    for (int j = -1; j <= 1; ++j) {
                        for (int i = -1; i <= 1; ++i) {
                            int nx = x + i;
                            int ny = y + j;
                            if (0 <= nx && nx < width &&
                                0 <= ny && ny < height &&
                                new_validity_mask->at(nx, ny, 0) == 255) {

                                float w = gauss[(j + 1) * 3 + (i + 1)];
                                norm += w;
                                value += (img->at(nx, ny, c) / 255.0f) * w;
                            }
                        }
                    }

                    if (norm == 0.0f)
                        continue;

                    now_valid = true;
                    img->at(x, y, c) = (value / norm) * 255.0f;
                }

                if (now_valid) {
                    local_valid_pixels.push_back(invalid_border_pixels[idx]);
                }
            }
            #pragma omp critical
            new_valid_pixels.insert(new_valid_pixels.end(),
                local_valid_pixels.begin(), local_valid_pixels.end());
        }

        /* This frontier has been fully consumed (turned into new_valid_pixels
         * or left permanently invalid); free its membership slots before
         * computing the next round's frontier. */
        for (std::pair<int, int> const & p : invalid_border_pixels) {
            in_frontier[p.second * width + p.first] = 0;
        }
        invalid_border_pixels.clear();

        /* Mark the new valid pixels valid in the validity mask. */
        for (std::size_t i = 0; i < new_valid_pixels.size(); ++i) {
             int x = new_valid_pixels[i].first;
             int y = new_valid_pixels[i].second;

             new_validity_mask->at(x, y, 0) = 255;
        }

        /* Calculate the set of invalid pixels at the border of the valid
         * area. Kept single-threaded: it's neighbor bookkeeping only (no
         * float math), much cheaper than the pass above, and two
         * new_valid_pixels can share an invalid neighbour, so deduping via
         * `in_frontier` needs to be serialized anyway. */
        for (std::size_t i = 0; i < new_valid_pixels.size(); ++i) {
            int x = new_valid_pixels[i].first;
            int y = new_valid_pixels[i].second;

            for (int j = -1; j <= 1; ++j) {
                 for (int i = -1; i <= 1; ++i) {
                     int nx = x + i;
                     int ny = y + j;
                     if (0 <= nx && nx < width &&
                         0 <= ny && ny < height &&
                         new_validity_mask->at(nx, ny, 0) == 0 &&
                         !in_frontier[ny * width + nx]) {

                         in_frontier[ny * width + nx] = 1;
                         invalid_border_pixels.push_back(std::pair<int, int>(nx, ny));
                    }
                }
            }
        }
    }
}

#ifdef MVSTEX_GPU
/* Only the byte-atlas path is ported to GPU (see edge_padding_gpu.h for
 * why); float/uint16_t atlases keep using the generic template above,
 * which always takes the CPU path regardless of use_gpu. */
template <>
void
TextureAtlas::apply_edge_padding<uint8_t>(bool use_gpu) {
    assert(image != NULL);
    assert(validity_mask != NULL);

    if (use_gpu) {
        mve::ByteImage::Ptr img = std::dynamic_pointer_cast<mve::ByteImage>(image);
        int const width = img->width();
        int const height = img->height();

        uint8_t const * pixel_ptr = &img->at(0, 0);
        std::vector<uint8_t> pixels(pixel_ptr,
            pixel_ptr + static_cast<std::size_t>(width) * height * 3);
        uint8_t const * mask_ptr = &validity_mask->at(0, 0);
        std::vector<uint8_t> mask(mask_ptr,
            mask_ptr + static_cast<std::size_t>(width) * height);

        tex::GPUEdgePadding gpu_padding(width, height, pixels, mask);
        if (gpu_padding.available()) {
            std::cout << "\tEdge padding GPU kernel ready." << std::endl;
            std::vector<uint8_t> result;
            if (gpu_padding.run(padding, &result)) {
                std::copy(result.begin(), result.end(), &img->at(0, 0));
                return;
            }
        }

        std::fprintf(stderr,
            "[mvstex GPU] edge padding GPU path unavailable/failed, "
            "falling back to CPU\n");
    }

    this->apply_edge_padding_cpu<uint8_t>();
}
#endif

struct VectorCompare {
    bool operator()(math::Vec2f const & lhs, math::Vec2f const & rhs) const {
        return lhs[0] < rhs[0] || (lhs[0] == rhs[0] && lhs[1] < rhs[1]);
    }
};

typedef std::map<math::Vec2f, std::size_t, VectorCompare> TexcoordMap;

void
TextureAtlas::merge_texcoords() {
    Texcoords tmp; tmp.swap(this->texcoords);

    TexcoordMap texcoord_map;
    for (math::Vec2f const & texcoord : tmp) {
        TexcoordMap::iterator iter = texcoord_map.find(texcoord);
        if (iter == texcoord_map.end()) {
            std::size_t texcoord_id = this->texcoords.size();
            texcoord_map[texcoord] = texcoord_id;
            this->texcoords.push_back(texcoord);
            this->texcoord_ids.push_back(texcoord_id);
        } else {
            this->texcoord_ids.push_back(iter->second);
        }
    }

}

void
TextureAtlas::finalize(bool use_gpu) {
    if (finalized) {
        throw util::Exception("TextureAtlas already finalized");
    }

    this->bin.reset();
    if (image->get_type() == mve::IMAGE_TYPE_FLOAT){
        this->apply_edge_padding<float>(use_gpu);
    }else if (image->get_type() == mve::IMAGE_TYPE_UINT16){
        this->apply_edge_padding<uint16_t>(use_gpu);
    }else{
        this->apply_edge_padding<uint8_t>(use_gpu);
    }
    this->validity_mask.reset();
    this->merge_texcoords();

    this->finalized = true;
}
