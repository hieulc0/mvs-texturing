/*
 * GPU-accelerated occlusion testing for calculate_data_costs.
 *
 * This header has no CUDA syntax in it on purpose: it's included from
 * calculate_data_costs.cpp, which is compiled by the regular C++ compiler,
 * not nvcc. All CUDA-specific code lives in visibility_kernel.cu, behind
 * the pimpl-style Impl pointer.
 *
 * Only built/linked when MVSTEX_GPU is enabled (see libs/tex/CMakeLists.txt).
 */
#ifndef TEX_VISIBILITY_GPU_HEADER
#define TEX_VISIBILITY_GPU_HEADER

#include <cstddef>
#include <cstdint>
#include <vector>

#include <math/vector.h>
#include <acc/bvh_tree.h>

#include "defines.h"

TEX_NAMESPACE_BEGIN

typedef acc::BVHTree<unsigned int, math::Vec3f> BVHTree;

/**
 * Uploads a CPU-built BVHTree to the GPU once, then answers batched
 * occlusion (any-hit) queries against it. Mirrors the boolean result of
 * BVHTree::intersect() -- "is there a triangle between [tmin, tmax] along
 * this ray" -- it does not report which triangle or where, since
 * calculate_face_projection_infos only ever uses the boolean.
 *
 * If GPU initialization fails for any reason (no device, OOM uploading the
 * tree, driver error), available() returns false and the caller is
 * expected to fall back to BVHTree::intersect() on the CPU -- this class
 * never throws.
 */
class GPUVisibilityTester {
public:
    /** One occlusion query: ray from `origin` along normalized `dir`,
     * tested against (tmax * kTminFactor, tmax). Matches the tmin/tmax
     * convention calculate_face_projection_infos already uses for its
     * CPU rays (see calculate_data_costs.cpp). */
    struct Ray {
        math::Vec3f origin;
        math::Vec3f dir;
        float tmax;
    };

    explicit GPUVisibilityTester(BVHTree const & bvh_tree);
    ~GPUVisibilityTester();

    bool available() const { return valid; }

    /* Fills *occluded (resized to rays.size()) with one bool per ray and
     * returns true on success. Returns false (and leaves *occluded
     * untouched) if !available() or if the batch failed on the device --
     * callers MUST check the return value and fall back to
     * BVHTree::intersect() on the CPU for that batch when it's false,
     * rather than assume any particular content in *occluded. */
    bool test_occlusion(std::vector<Ray> const & rays,
        std::vector<bool> * occluded) const;

private:
    GPUVisibilityTester(GPUVisibilityTester const &) = delete;
    GPUVisibilityTester & operator=(GPUVisibilityTester const &) = delete;

    struct Impl;
    Impl * impl;
    bool valid;
};

TEX_NAMESPACE_END

#endif /* TEX_VISIBILITY_GPU_HEADER */
