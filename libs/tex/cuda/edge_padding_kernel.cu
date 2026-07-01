/*
 * CUDA implementation backing GPUEdgePadding (edge_padding_gpu.h).
 *
 * Direct port of the CPU dilation loop in
 * TextureAtlas::apply_edge_padding<uint8_t>() (texture_atlas.cpp), done as
 * a dense whole-image kernel per round instead of porting the CPU path's
 * std::set-based frontier bookkeeping -- see edge_padding_gpu.h for why
 * that's a legitimate simplification, not an approximation. Loop order and
 * weight values inside a channel match the CPU version pixel-for-pixel so
 * results should be numerically identical modulo floating point
 * reassociation from compiler/ISA differences (the /16 Gaussian weights
 * are exact binary fractions either way, so there isn't even a rounding
 * difference from that).
 *
 * Pixel/mask buffers are mutated in place across rounds: a thread only
 * writes to its own (currently-invalid) pixel and only reads neighbours
 * that `mask_in` marks valid for this round, and `mask_in` isn't mutated
 * until the *next* round's buffer -- same "no pixel a thread might read
 * is written by another thread this round" invariant the CPU OpenMP
 * rewrite relies on (see texture_atlas.cpp), just enforced by ping-ponging
 * mask buffers instead of an omp critical merge.
 */
#include "edge_padding_gpu.h"

#include <cstdio>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

TEX_NAMESPACE_BEGIN

namespace {

/* Matches the CPU's `gauss /= 16.0f` step element-for-element -- all of
 * 1/16, 2/16, 4/16 are exact binary fractions, so precomputing here vs.
 * dividing at each use in the CPU code makes no numeric difference. */
__device__ const float kGauss[9] = {
    1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f,
    2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f,
    1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f
};

__global__ void edgePaddingRoundKernel(uint8_t * pixels, uint8_t const * mask_in,
        uint8_t * mask_out, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = width * height;
    if (idx >= n) return;

    if (mask_in[idx] == 255) {
        mask_out[idx] = 255;
        return;
    }

    int x = idx % width;
    int y = idx / width;
    bool now_valid = false;

    for (int c = 0; c < 3; ++c) {
        float norm = 0.0f;
        float value = 0.0f;
        for (int j = -1; j <= 1; ++j) {
            int ny = y + j;
            if (ny < 0 || ny >= height) continue;
            for (int i = -1; i <= 1; ++i) {
                int nx = x + i;
                if (nx < 0 || nx >= width) continue;
                int nidx = ny * width + nx;
                if (mask_in[nidx] != 255) continue;

                float w = kGauss[(j + 1) * 3 + (i + 1)];
                norm += w;
                value += (pixels[nidx * 3 + c] / 255.0f) * w;
            }
        }

        if (norm == 0.0f) continue;
        now_valid = true;
        pixels[idx * 3 + c] = (uint8_t)((value / norm) * 255.0f);
    }

    mask_out[idx] = now_valid ? 255 : 0;
}

#define MVSTEX_CUDA_CHECK(expr) \
    do { \
        cudaError_t _err = (expr); \
        if (_err != cudaSuccess) { \
            std::fprintf(stderr, "[mvstex GPU] %s failed: %s (%s:%d)\n", \
                #expr, cudaGetErrorString(_err), __FILE__, __LINE__); \
            return false; \
        } \
    } while (0)

} // namespace

struct GPUEdgePadding::Impl {
    uint8_t * d_pixels = nullptr;
    uint8_t * d_mask_a = nullptr;
    uint8_t * d_mask_b = nullptr;
    int width = 0;
    int height = 0;
    std::size_t num_pixels = 0;

    bool upload(int w, int h, std::vector<uint8_t> const & pixels,
            std::vector<uint8_t> const & mask) {
        width = w;
        height = h;
        num_pixels = static_cast<std::size_t>(w) * h;
        if (num_pixels == 0) return false;
        if (pixels.size() != num_pixels * 3 || mask.size() != num_pixels) return false;

        MVSTEX_CUDA_CHECK(cudaMalloc(&d_pixels, num_pixels * 3));
        MVSTEX_CUDA_CHECK(cudaMalloc(&d_mask_a, num_pixels));
        MVSTEX_CUDA_CHECK(cudaMalloc(&d_mask_b, num_pixels));
        MVSTEX_CUDA_CHECK(cudaMemcpy(d_pixels, pixels.data(), num_pixels * 3,
            cudaMemcpyHostToDevice));
        MVSTEX_CUDA_CHECK(cudaMemcpy(d_mask_a, mask.data(), num_pixels,
            cudaMemcpyHostToDevice));
        return true;
    }

    ~Impl() {
        if (d_pixels) cudaFree(d_pixels);
        if (d_mask_a) cudaFree(d_mask_a);
        if (d_mask_b) cudaFree(d_mask_b);
    }
};

GPUEdgePadding::GPUEdgePadding(int width, int height,
        std::vector<uint8_t> const & pixels, std::vector<uint8_t> const & validity_mask) {
    impl = new Impl();
    valid = impl->upload(width, height, pixels, validity_mask);
    if (!valid) {
        delete impl;
        impl = nullptr;
    }
}

GPUEdgePadding::~GPUEdgePadding() {
    delete impl;
}

bool GPUEdgePadding::run(unsigned int padding, std::vector<uint8_t> * out_pixels) const {
    if (!valid) return false;

    uint8_t * mask_in = impl->d_mask_a;
    uint8_t * mask_out = impl->d_mask_b;

    const int blockSize = 256;
    const int numBlocks = (int)((impl->num_pixels + blockSize - 1) / blockSize);

    for (unsigned int n = 0; n <= padding; ++n) {
        edgePaddingRoundKernel<<<numBlocks, blockSize>>>(impl->d_pixels,
            mask_in, mask_out, impl->width, impl->height);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::fprintf(stderr, "[mvstex GPU] edge padding round %u failed: %s\n",
                n, cudaGetErrorString(err));
            return false;
        }
        std::swap(mask_in, mask_out);
    }

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[mvstex GPU] edge padding sync failed: %s\n",
            cudaGetErrorString(err));
        return false;
    }

    out_pixels->resize(impl->num_pixels * 3);
    err = cudaMemcpy(out_pixels->data(), impl->d_pixels, impl->num_pixels * 3,
        cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[mvstex GPU] edge padding download failed: %s\n",
            cudaGetErrorString(err));
        return false;
    }
    return true;
}

TEX_NAMESPACE_END
