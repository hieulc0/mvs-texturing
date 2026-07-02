/*
 * Copyright (C) 2015, Nils Moehrle
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include <cstdint>
#include <iostream>
#include <memory>

#include <math/vector.h>
#include <util/timer.h>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>

#include "poisson_blending.h"

typedef Eigen::SparseMatrix<float> SpMat;

double poisson_blend_build_time_sec = 0.0;
double poisson_blend_factorize_time_sec = 0.0;
double poisson_blend_solve_time_sec = 0.0;
double poisson_blend_fallback_count = 0.0;

math::Vec3f simple_laplacian(int i, mve::FloatImage::ConstPtr img){
    const int width = img->width();
    assert(i > width + 1 && i < img->get_pixel_amount() - width -1);

    return -4.0f * math::Vec3f(&img->at(i, 0))
        + math::Vec3f(&img->at(i - width, 0))
        + math::Vec3f(&img->at(i - 1, 0))
        + math::Vec3f(&img->at(i + 1, 0))
        + math::Vec3f(&img->at(i + width, 0));
}

bool valid_mask(mve::ByteImage::ConstPtr mask){
    const int width = mask->width();
    const int height = mask->height();

    for (int x = 0; x < width; ++x)
        if (mask->at(x, 0, 0) == 255 || mask->at(x, height - 1, 0) == 255)
            return false;

    for (int y = 0; y < height; ++y)
        if (mask->at(0, y, 0) == 255 || mask->at(width - 1, y, 0) == 255)
            return false;

    //TODO check for sane boundary conditions...

    return true;
}

void
poisson_blend(mve::FloatImage::ConstPtr src, mve::ByteImage::ConstPtr mask,
    mve::FloatImage::Ptr dest, float alpha) {

    assert(src->width() == mask->width() && mask->width() == dest->width());
    assert(src->height() == mask->height() && mask->height() == dest->height());
    assert(src->channels() == 3 && dest->channels() == 3);
    assert(mask->channels() == 1);
    assert(valid_mask(mask));

    const int n = dest->get_pixel_amount();
    const int width = dest->width();
    const int height = dest->height();
    const int channels = dest->channels();

    util::WallTimer build_timer;

    /* Reformulated system (docs/gpu-accel-texturing.md §21/§22): the
     * mask==128|64 rows are pure Dirichlet constraints (identity row,
     * RHS = the already-known dest value) -- they were never actually
     * unknowns, just copied straight back out unchanged. Eliminating them
     * from the unknown vector and folding their known values into the
     * interior (mask==255) rows' RHS leaves a system over interior pixels
     * only, whose matrix is the discrete Laplacian's negation: this is
     * symmetric positive definite (unlike the original mixed system,
     * which is non-symmetric). A first attempt solved this with
     * Eigen::ConjugateGradient -- mathematically valid and converged in
     * ~70 iterations/solve (not near the iteration cap), but measured
     * ~27x slower per solve than SparseLU regardless (§25): with ~7700
     * patches x 3 channels of genuinely tiny systems, SparseLU's shape
     * (one factorization reused cheaply across 3 channel solves) beats
     * any iterative method's fixed per-iteration overhead, which is paid
     * fresh on every one of ~23000 solves with nothing to amortize.
     * Eigen::SimplicialLDLT keeps that same amortize-across-channels
     * shape as SparseLU while still being a direct Cholesky-family
     * solver that exploits the symmetry this reformulation unlocked --
     * see docs/gpu-accel-texturing.md §25 for the measured comparison. */
    mve::Image<int>::Ptr indices = mve::Image<int>::create(width, height, 1);
    indices->fill(-1);
    int index = 0;
    for (int i = 0; i < n; ++i) {
        if (mask->at(i) == 255) {
            indices->at(i) = index;
            index += 1;
        }
    }
    const int nnz = index;

    std::vector<math::Vec3f> coefficients_b;
    coefficients_b.resize(nnz);

    std::vector<Eigen::Triplet<float, int> > coefficients_A;
    coefficients_A.reserve(nnz * 5);

    for (int i = 0; i < n; ++i) {
        if (mask->at(i) != 255) continue;
        const int row = indices->at(i);

        const int neighbors[4] = { i - width, i - 1, i + 1, i + width };

        math::Vec3f l_d = simple_laplacian(i, dest);
        math::Vec3f l_s = simple_laplacian(i, src);
        math::Vec3f b_i = alpha * l_s + (1.0f - alpha) * l_d;

        /* Diagonal: +4 is the negated discrete Laplacian's diagonal
         * (original was -4; negating both sides of the equation is what
         * makes the matrix positive- rather than negative-definite). */
        coefficients_A.push_back(Eigen::Triplet<float, int>(row, row, 4.0f));

        for (int k = 0; k < 4; ++k) {
            const int j = neighbors[k];
            const uint8_t neighbor_mask = mask->at(j);
            /* All neighbours should be eighter border conditions or part of the optimization. */
            assert(neighbor_mask == 255 || neighbor_mask == 128 || neighbor_mask == 64);
            if (neighbor_mask == 255) {
                coefficients_A.push_back(Eigen::Triplet<float, int>(row, indices->at(j), -1.0f));
            } else {
                /* Known Dirichlet value -- move it to the RHS instead of
                 * keeping it as an identity row/column in A. */
                b_i -= math::Vec3f(&dest->at(j, 0));
            }
        }

        coefficients_b[row] = -b_i;
    }

    SpMat A(nnz, nnz);
    A.setFromTriplets(coefficients_A.begin(), coefficients_A.end());

    double build_elapsed = build_timer.get_elapsed_sec();
    #pragma omp atomic
    poisson_blend_build_time_sec += build_elapsed;

    util::WallTimer factorize_timer;
    Eigen::SimplicialLDLT<SpMat, Eigen::Lower> solver;
    solver.compute(A);
    double factorize_elapsed = factorize_timer.get_elapsed_sec();
    #pragma omp atomic
    poisson_blend_factorize_time_sec += factorize_elapsed;

    /* Exact fallback, lazily constructed and only paid for if LDLT
     * actually reports a numerical issue for this patch (should be rare
     * given A is genuinely SPD, but floating-point roundoff on a
     * degenerate/tiny patch is cheap insurance against, not expected to
     * fire in the common case) -- same bounding-the-risk pattern as
     * every non-bit-identical change in this function so far. */
    std::unique_ptr<Eigen::SparseLU<SpMat, Eigen::COLAMDOrdering<int> > > fallback_solver;

    util::WallTimer solve_timer;
    for (int channel = 0; channel < channels; ++channel) {
        Eigen::VectorXf b(nnz);
        for (int i = 0; i < nnz; ++i)
            b[i] = coefficients_b[i][channel];

        Eigen::VectorXf x(nnz);
        bool need_fallback = (solver.info() != Eigen::Success);
        if (!need_fallback) {
            x = solver.solve(b);
            need_fallback = (solver.info() != Eigen::Success);
        }
        if (need_fallback) {
            if (!fallback_solver) {
                fallback_solver.reset(new Eigen::SparseLU<SpMat, Eigen::COLAMDOrdering<int> >());
                fallback_solver->compute(A);
            }
            x = fallback_solver->solve(b);
            #pragma omp atomic
            poisson_blend_fallback_count += 1;
        }

        for (int i = 0; i < n; ++i) {
            int idx = indices->at(i);
            if (idx != -1) dest->at(i, channel) = x[idx];
        }
    }
    double solve_elapsed = solve_timer.get_elapsed_sec();
    #pragma omp atomic
    poisson_blend_solve_time_sec += solve_elapsed;
}
