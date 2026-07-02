/*
 * Copyright (C) 2015, Nils Moehrle
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef TEX_TEXTUREATLAS_HEADER
#define TEX_TEXTUREATLAS_HEADER


#include <vector>

#include <util/exception.h>
#include <math/vector.h>
#include <mve/mesh.h>
#include <mve/image.h>

#include "tri.h"
#include "texture_patch.h"
#include "rectangular_bin.h"

/**
  * Class representing a texture atlas.
  */
class TextureAtlas {
    public:
        typedef std::shared_ptr<TextureAtlas> Ptr;

        typedef std::vector<std::size_t> Faces;
        typedef std::vector<std::size_t> TexcoordIds;
        typedef std::vector<math::Vec2f> Texcoords;

    private:
        unsigned int const size;
        unsigned int const padding;
        bool finalized;
        bool grayscale;

        Faces faces;
        Texcoords texcoords;
        TexcoordIds texcoord_ids;

        mve::ImageBase::Ptr image;
        mve::ByteImage::Ptr validity_mask;

        RectangularBin::Ptr bin;

        template <typename T>
        void apply_edge_padding(bool use_gpu);

        template <typename T>
        void apply_edge_padding_cpu(void);

        void merge_texcoords(void);

    public:
        TextureAtlas(unsigned int size, mve::ImageType type, bool grayscale);

        static TextureAtlas::Ptr create(unsigned int size, mve::ImageType type, bool grayscale);

        Faces const & get_faces(void) const;
        TexcoordIds const & get_texcoord_ids(void) const;
        Texcoords const & get_texcoords(void) const;
        mve::ImageBase::Ptr get_image(void) const;

        /* Serial bin-fit decision only -- must be called in placement order
         * within a given atlas, since each call depends on every prior
         * placement. On success, fills *rect with the placed position and
         * returns true; the caller must then call commit() with the same
         * rect to actually write the patch's pixels/bookkeeping. See
         * docs/gpu-accel-texturing.md §33 Track A. */
        bool try_place(TexturePatch::ConstPtr texture_patch, Rect<int> * rect);

        /* The copy_into pixel copies + faces/texcoords bookkeeping half of
         * the old insert(). Writes to a disjoint image region per call (safe
         * to run concurrently across patches), but appends to the shared
         * faces/texcoords vectors under a critical section. Safe to call
         * from an `#pragma omp parallel for` once try_place() has already
         * reserved rect for this patch. */
        void commit(TexturePatch::ConstPtr texture_patch, Rect<int> const & rect);

        /* Thin serial wrapper: try_place() + commit(). Kept for callers that
         * don't need the parallel commit split. */
        bool insert(TexturePatch::ConstPtr texture_patch);

        /* `use_gpu` only takes effect for byte (8-bit PNG) atlases built
         * with MVSTEX_GPU -- see apply_edge_padding's uint8_t specialization
         * in texture_atlas.cpp. Ignored (and safe to pass) otherwise. */
        void finalize(bool use_gpu = false);
        bool is_grayscale();
};

/* Diagnostic-only globals, see docs/gpu-accel-texturing.md §31/§33 --
 * wall-clock time summed across all TextureAtlas::try_place()/commit()
 * calls (formerly two phases of a single insert()), split between the
 * serial bin-packing decision (RectangularBin::insert, which must stay
 * serial -- each call depends on every prior placement in the same
 * atlas) and the pixel-copy + bookkeeping work (copy_into calls +
 * faces/texcoords bookkeeping, which write to disjoint per-patch regions
 * and are parallelized via commit() as of §33 Track A). The bin-fit timer
 * is only ever updated from the serial placement loop, so plain += stays
 * safe there; the copy+bookkeeping timer is now updated from parallel
 * commit() calls, so it uses #pragma omp atomic. Not safe to *read*
 * while any try_place()/commit() call may still be in flight. */
extern double atlas_insert_binfit_time_sec;
extern double atlas_insert_copy_time_sec;

inline TextureAtlas::Ptr
TextureAtlas::create(unsigned int size, mve::ImageType type, bool grayscale) {
    return Ptr(new TextureAtlas(size, type, grayscale));
}

inline TextureAtlas::Faces const &
TextureAtlas::get_faces(void) const {
    return faces;
}

inline TextureAtlas::TexcoordIds const &
TextureAtlas::get_texcoord_ids(void) const {
    return texcoord_ids;
}

inline TextureAtlas::Texcoords const &
TextureAtlas::get_texcoords(void) const {
    return texcoords;
}

inline mve::ImageBase::Ptr
TextureAtlas::get_image(void) const {
    if (!finalized) {
        throw util::Exception("Texture atlas not finalized");
    }
    return image;
}

inline bool
TextureAtlas::is_grayscale(){
    return grayscale;
}

#endif /* TEX_TEXTUREATLAS_HEADER */
