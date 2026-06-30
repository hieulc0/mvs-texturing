/*
 * CUDA implementation backing GPUVisibilityTester (visibility_gpu.h).
 *
 * Reuses the CPU-built acc::BVHTree topology (already a flat, SAH-built
 * binary tree -- see BVHTree::get_nodes()/get_tris() added in the rayint
 * fork) instead of building a tree on the device. Only ray-mesh traversal
 * moves to the GPU; tree construction stays on the CPU where it already
 * runs once per texturing run and isn't the bottleneck.
 *
 * The AABB slab test and triangle intersection below are direct ports of
 * acc::intersect() in elibs/rayint/libs/acc/primitives.h -- same epsilons,
 * same algorithm -- so GPU results should match the CPU path exactly
 * (mod floating point reassociation from compiler/ISA differences). Do not
 * build this file with --use_fast_math: it relies on IEEE-754 inf/nan
 * behavior from division by zero in the slab test, same as the CPU code.
 *
 * This only answers "is this ray occluded" (any-hit), not "what did it
 * hit" -- calculate_face_projection_infos only ever needs the boolean, so
 * traversal can stop at the first intersecting triangle instead of
 * tracking the closest one.
 */
#include "visibility_gpu.h"

#include <cstdio>
#include <vector>

#include <cuda_runtime.h>
#include <math_constants.h>

TEX_NAMESPACE_BEGIN

namespace {

constexpr int kBVHStackSize = 64; // see note in docs/gpu-accel-texturing.md
constexpr float kTminFactor = 0.0001f; // matches the CPU ray.tmin convention
constexpr unsigned int kNAI = 0xFFFFFFFFu; // BVHTree::nai() for IdxType = unsigned int

struct GPUNode {
    float3 aabb_min;
    float3 aabb_max;
    uint32_t left;
    uint32_t right;
    uint32_t first;
    uint32_t last;
};

struct GPUTri {
    float3 a;
    float3 b;
    float3 c;
};

struct GPURay {
    float3 origin;
    float3 dir;
    float tmax;
};

__device__ inline float3 vsub(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline float3 vadd(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ inline float3 vscale(float3 a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ inline float vdot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline float3 vcross(float3 a, float3 b) {
    return make_float3(a.y * b.z - a.z * b.y,
                        a.z * b.x - a.x * b.z,
                        a.x * b.y - a.y * b.x);
}

__device__ inline float vnorm(float3 a) {
    return sqrtf(vdot(a, a));
}

/* Port of acc::intersect(Ray, AABB, float*) -- Tavian Barnes' branchless
 * slab test. Relies on IEEE-754 division-by-zero -> +-inf semantics for
 * axis-aligned ray directions, exactly like the CPU version. */
__device__ inline bool intersectAABB(float3 origin, float3 dir, float tmin_in,
        float tmax_in, float3 bmin, float3 bmax, float * tmin_out) {
    float tmin = tmin_in, tmax = tmax_in;
    const float inf = CUDART_INF_F;

    float t1 = (bmin.x - origin.x) / dir.x;
    float t2 = (bmax.x - origin.x) / dir.x;
    tmin = fmaxf(tmin, fminf(fminf(t1, t2), inf));
    tmax = fminf(tmax, fmaxf(fmaxf(t1, t2), -inf));

    t1 = (bmin.y - origin.y) / dir.y;
    t2 = (bmax.y - origin.y) / dir.y;
    tmin = fmaxf(tmin, fminf(fminf(t1, t2), inf));
    tmax = fminf(tmax, fmaxf(fmaxf(t1, t2), -inf));

    t1 = (bmin.z - origin.z) / dir.z;
    t2 = (bmax.z - origin.z) / dir.z;
    tmin = fmaxf(tmin, fminf(fminf(t1, t2), inf));
    tmax = fminf(tmax, fmaxf(fmaxf(t1, t2), -inf));

    *tmin_out = tmin;
    return tmax >= fmaxf(tmin, 0.0f);
}

/* Port of acc::intersect(Ray, Tri, float*, Vec3f*). Cosine/denom kept in
 * double, matching the CPU version, to stay numerically close to it. */
__device__ inline bool intersectTri(float3 origin, float3 dir, float tmin,
        float tmax, GPUTri const & tri) {
    const float eps = 1e-3f;

    float3 v0 = vsub(tri.b, tri.a);
    float3 v1 = vsub(tri.c, tri.a);
    float3 normal = vcross(v1, v0);
    float nlen = vnorm(normal);
    if (nlen < 1.1920929e-7f /* FLT_EPSILON */) return false;
    normal = vscale(normal, 1.0f / nlen);

    double cosine = (double)vdot(normal, dir);
    if (fabs(cosine) < 2.2204460492503131e-16 /* DBL_EPSILON */) return false;

    float t = (float)(-(double)vdot(normal, vsub(origin, tri.a)) / cosine);
    if (t < tmin || tmax < t) return false;

    float3 v2 = vadd(vsub(origin, tri.a), vscale(dir, t));

    float d00 = vdot(v0, v0);
    float d01 = vdot(v0, v1);
    float d11 = vdot(v1, v1);
    float d20 = vdot(v2, v0);
    float d21 = vdot(v2, v1);
    double denom = (double)d00 * d11 - (double)d01 * d01;

    double b1 = (d11 * d20 - d01 * d21) / denom;
    double b2 = (d00 * d21 - d01 * d20) / denom;
    double b0 = 1.0 - b1 - b2;

    return -eps <= b0 && b0 <= 1.0 + eps
        && -eps <= b1 && b1 <= 1.0 + eps
        && -eps <= b2 && b2 <= 1.0 + eps;
}

/* Any-hit traversal: same iterative stack-based walk as
 * BVHTree::intersect(), but stops at the first intersecting triangle
 * instead of tracking the closest one (calculate_face_projection_infos
 * only needs hit/no-hit). Node order on the stack doesn't matter for
 * correctness here, only for closest-hit early-out, which we don't need. */
__device__ bool anyHit(GPUNode const * nodes, GPUTri const * tris,
        float3 origin, float3 dir, float tmin, float tmax) {
    uint32_t stack[kBVHStackSize];
    int sp = 0;
    stack[sp++] = 0; // root

    while (sp > 0) {
        uint32_t node_id = stack[--sp];
        GPUNode node = nodes[node_id];

        if (node.left != kNAI && node.right != kNAI) {
            float tl, tr;
            GPUNode const & ln = nodes[node.left];
            GPUNode const & rn = nodes[node.right];
            bool hitL = intersectAABB(origin, dir, tmin, tmax, ln.aabb_min, ln.aabb_max, &tl);
            bool hitR = intersectAABB(origin, dir, tmin, tmax, rn.aabb_min, rn.aabb_max, &tr);
            // Stack overflow is only reachable with a pathologically
            // unbalanced tree; bail out conservatively (treat as occluded)
            // rather than overflow the local array.
            if (sp + 2 > kBVHStackSize) return true;
            if (hitL) stack[sp++] = node.left;
            if (hitR) stack[sp++] = node.right;
        } else {
            for (uint32_t i = node.first; i < node.last; ++i) {
                if (intersectTri(origin, dir, tmin, tmax, tris[i])) return true;
            }
        }
    }
    return false;
}

__global__ void occlusionKernel(GPUNode const * nodes, GPUTri const * tris,
        GPURay const * rays, uint8_t * results, int numRays) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numRays) return;

    GPURay ray = rays[i];
    float tmin = ray.tmax * kTminFactor;
    results[i] = anyHit(nodes, tris, ray.origin, ray.dir, tmin, ray.tmax) ? 1 : 0;
}

inline float3 toFloat3(math::Vec3f const & v) {
    return make_float3(v[0], v[1], v[2]);
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

struct GPUVisibilityTester::Impl {
    GPUNode * d_nodes = nullptr;
    GPUTri * d_tris = nullptr;
    std::size_t num_nodes = 0;
    std::size_t num_tris = 0;

    bool upload(BVHTree const & bvh_tree) {
        auto const & nodes = bvh_tree.get_nodes();
        auto const & tris = bvh_tree.get_tris();
        num_nodes = nodes.size();
        num_tris = tris.size();

        if (num_nodes == 0 || num_tris == 0) return false;

        std::vector<GPUNode> flat_nodes(num_nodes);
        for (std::size_t i = 0; i < num_nodes; ++i) {
            auto const & n = nodes[i];
            flat_nodes[i].aabb_min = toFloat3(n.aabb.min);
            flat_nodes[i].aabb_max = toFloat3(n.aabb.max);
            flat_nodes[i].left = (uint32_t)n.left;
            flat_nodes[i].right = (uint32_t)n.right;
            flat_nodes[i].first = (uint32_t)n.first;
            flat_nodes[i].last = (uint32_t)n.last;
        }

        std::vector<GPUTri> flat_tris(num_tris);
        for (std::size_t i = 0; i < num_tris; ++i) {
            auto const & t = tris[i];
            flat_tris[i].a = toFloat3(t.a);
            flat_tris[i].b = toFloat3(t.b);
            flat_tris[i].c = toFloat3(t.c);
        }

        MVSTEX_CUDA_CHECK(cudaMalloc(&d_nodes, num_nodes * sizeof(GPUNode)));
        MVSTEX_CUDA_CHECK(cudaMalloc(&d_tris, num_tris * sizeof(GPUTri)));
        MVSTEX_CUDA_CHECK(cudaMemcpy(d_nodes, flat_nodes.data(),
            num_nodes * sizeof(GPUNode), cudaMemcpyHostToDevice));
        MVSTEX_CUDA_CHECK(cudaMemcpy(d_tris, flat_tris.data(),
            num_tris * sizeof(GPUTri), cudaMemcpyHostToDevice));
        return true;
    }

    ~Impl() {
        if (d_nodes) cudaFree(d_nodes);
        if (d_tris) cudaFree(d_tris);
    }
};

GPUVisibilityTester::GPUVisibilityTester(BVHTree const & bvh_tree) {
    impl = new Impl();
    valid = impl->upload(bvh_tree);
    if (!valid) {
        delete impl;
        impl = nullptr;
    }
}

GPUVisibilityTester::~GPUVisibilityTester() {
    delete impl;
}

bool GPUVisibilityTester::test_occlusion(std::vector<Ray> const & rays,
        std::vector<bool> * occluded) const {
    if (!valid) return false;
    if (rays.empty()) {
        occluded->clear();
        return true;
    }

    std::size_t n = rays.size();
    std::vector<GPURay> host_rays(n);
    for (std::size_t i = 0; i < n; ++i) {
        host_rays[i].origin = toFloat3(rays[i].origin);
        host_rays[i].dir = toFloat3(rays[i].dir);
        host_rays[i].tmax = rays[i].tmax;
    }

    GPURay * d_rays = nullptr;
    uint8_t * d_results = nullptr;
    std::vector<uint8_t> host_results(n, 0);

    bool ok = true;
    cudaError_t err;

    err = cudaMalloc(&d_rays, n * sizeof(GPURay));
    ok = ok && (err == cudaSuccess);
    if (ok) {
        err = cudaMalloc(&d_results, n * sizeof(uint8_t));
        ok = ok && (err == cudaSuccess);
    }
    if (ok) {
        err = cudaMemcpy(d_rays, host_rays.data(), n * sizeof(GPURay),
            cudaMemcpyHostToDevice);
        ok = ok && (err == cudaSuccess);
    }
    if (ok) {
        const int blockSize = 256;
        const int numBlocks = (int)((n + blockSize - 1) / blockSize);
        occlusionKernel<<<numBlocks, blockSize>>>(impl->d_nodes, impl->d_tris,
            d_rays, d_results, (int)n);
        err = cudaGetLastError();
        ok = ok && (err == cudaSuccess);
    }
    if (ok) {
        err = cudaMemcpy(host_results.data(), d_results, n * sizeof(uint8_t),
            cudaMemcpyDeviceToHost);
        ok = ok && (err == cudaSuccess);
    }

    if (d_rays) cudaFree(d_rays);
    if (d_results) cudaFree(d_results);

    if (!ok) {
        std::fprintf(stderr, "[mvstex GPU] occlusion batch failed: %s\n",
            cudaGetErrorString(err));
        return false; // leave *occluded untouched; caller must check for that
    }

    occluded->resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        (*occluded)[i] = host_results[i] != 0;
    }
    return true;
}

TEX_NAMESPACE_END
