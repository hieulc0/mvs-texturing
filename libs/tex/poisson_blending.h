/*
 * Copyright (C) 2015, Nils Moehrle
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef TEX_POISSONBLENDING_HEADER
#define TEX_POISSONBLENDING_HEADER

#include "mve/image.h"

void
poisson_blend(mve::FloatImage::ConstPtr src, mve::ByteImage::ConstPtr mask,
    mve::FloatImage::Ptr dest, float alpha);

/* Diagnostic-only globals, see docs/gpu-accel-texturing.md §18/§19 --
 * wall-clock time summed across all threads/calls, split by which part of
 * poisson_blend() it was spent in, to find what actually dominates its
 * cost (972ms of "Blending texture patches"' ~979ms of core-work per
 * §17/§18) before changing anything about it. Not safe to *read* while
 * any poisson_blend() call may still be in flight -- only meant to be
 * read once every call for a given local_seam_leveling() run has
 * finished. */
extern double poisson_blend_build_time_sec;
extern double poisson_blend_factorize_time_sec;
extern double poisson_blend_solve_time_sec;

/* How many times poisson_blend() fell back from the iterative BiCGSTAB
 * solve to the exact SparseLU one (see docs/gpu-accel-texturing.md
 * §19-20) -- expected to stay near 0. */
extern long poisson_blend_fallback_count;

#endif /* TEX_POISSONBLENDING_HEADER */
