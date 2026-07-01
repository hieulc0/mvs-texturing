/*
 * GPU-accelerated edge padding (border dilation) for texture atlases.
 *
 * This header has no CUDA syntax in it on purpose: it's included from
 * texture_atlas.cpp, which is compiled by the regular C++ compiler, not
 * nvcc. All CUDA-specific code lives in edge_padding_kernel.cu, behind the
 * pimpl-style Impl pointer -- mirrors visibility_gpu.h's structure.
 *
 * Only built/linked when MVSTEX_GPU is enabled (see libs/tex/CMakeLists.txt).
 *
 * Scope: only the byte (mve::IMAGE_TYPE_UINT8) atlas path is ported -- the
 * common case for ODM's default 8-bit PNG atlases. FLOAT/UINT16 atlases
 * (HDR/16-bit workflows) still go through the CPU path in
 * TextureAtlas::apply_edge_padding's generic template; see
 * docs/gpu-accel-texturing.md for why this was scoped down rather than
 * templating the kernel over all three types.
 */
#ifndef TEX_EDGE_PADDING_GPU_HEADER
#define TEX_EDGE_PADDING_GPU_HEADER

#include <cstdint>
#include <vector>

#include "../defines.h"

TEX_NAMESPACE_BEGIN

/**
 * Uploads an atlas's RGB byte image + validity mask once, then runs the
 * border-dilation rounds (`TextureAtlas::apply_edge_padding`'s CPU loop)
 * entirely on the device before downloading the result.
 *
 * The dilation is done as a dense whole-image pass per round rather than
 * porting the CPU path's frontier-set bookkeeping: since a pixel with no
 * valid neighbour just computes a zero-weight no-op, checking every pixel
 * every round is mathematically identical to only checking the frontier,
 * not an approximation -- see docs/gpu-accel-texturing.md for the
 * bit-for-bit equivalence check this was validated against. That makes the
 * device side a plain masked-diffusion kernel, no queues/atomics needed.
 *
 * If GPU initialization fails for any reason (no device, OOM, driver
 * error), available() returns false and the caller is expected to fall
 * back to the CPU path -- this class never throws.
 */
class GPUEdgePadding {
public:
    /* `pixels` must be width*height*3 bytes (interleaved RGB, row-major --
     * the same flat layout as `&mve::ByteImage::at(0, 0)`). `validity_mask`
     * must be width*height bytes, each either 0 or 255. */
    GPUEdgePadding(int width, int height,
        std::vector<uint8_t> const & pixels,
        std::vector<uint8_t> const & validity_mask);
    ~GPUEdgePadding();

    bool available() const { return valid; }

    /* Runs `padding + 1` dilation rounds (matching
     * `for (unsigned int n = 0; n <= padding; ++n)` in the CPU path) and
     * fills *out_pixels (resized to width*height*3) with the result.
     * Returns false (and leaves *out_pixels untouched) if !available() or
     * if a round failed on the device -- caller MUST check the return
     * value and fall back to the CPU path when it's false. */
    bool run(unsigned int padding, std::vector<uint8_t> * out_pixels) const;

private:
    GPUEdgePadding(GPUEdgePadding const &) = delete;
    GPUEdgePadding & operator=(GPUEdgePadding const &) = delete;

    struct Impl;
    Impl * impl;
    bool valid;
};

TEX_NAMESPACE_END

#endif /* TEX_EDGE_PADDING_GPU_HEADER */
