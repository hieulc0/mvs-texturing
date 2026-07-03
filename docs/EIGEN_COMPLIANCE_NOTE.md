# Engineering Note: `Eigen::SparseCholesky` in `poisson_blending.cpp`

## Context

`libs/tex/poisson_blending.cpp` uses `Eigen::SimplicialLDLT` with its
default `Eigen::AMDOrdering`, both from Eigen's `SparseCholesky` module.
This project's top-level `CMakeLists.txt:19` sets
`add_definitions(-DEIGEN_MPL2_ONLY)` for the whole build, and the
vendored Eigen (pinned at 3.3.2, fetched by `elibs/CMakeLists.txt`) has
not relicensed `SparseCholesky` as MPL2 — it stays LGPL until Eigen 3.4+.
Compiling this file with `-DEIGEN_MPL2_ONLY` active fails outright:
`#error The SparseCholesky module has nothing to offer in MPL2 only mode`.

An earlier version of this note recommended either upgrading vendored
Eigen to 3.4+, or swapping to `Eigen::ConjugateGradient`. Both were tried
for real and rejected: the iterative solver measured ~2.2–27x slower for
this workload's shape (many tiny independent systems — see the parent
project's `docs/gpu-accel-texturing.md` §25), and a 5-year Eigen version
bump was judged too risky to validate project-wide from a workspace with
no way to run the full pipeline. Neither is revisited here.

## Why removing the restriction for this one file is safe

LGPL's substantive obligation is that the combined work's source stays
available under LGPL-compatible terms (or, for a library, that a user can
relink against a modified version of it). It does **not** require
relicensing the surrounding program to LGPL, unlike GPL. This fork is
consumed by WebODM/ODX, which is licensed AGPL-3.0 at the top level — a
strictly *stronger* copyleft than LGPL, whose source-availability
requirement (including over-the-network use) already exceeds what LGPL
asks for. So embedding `SparseCholesky` here needs no license change at
the WebODM/ODX project level. This was confirmed against Eigen's own
FAQ/mailing list guidance, not assumed.

**This pass-through only holds while this file is consumed by a project
under an equivalent-or-stronger copyleft license.** If you clone or reuse
this fork standalone, or embed it in a project that is not itself
source-available under LGPL-compatible terms, you are responsible for
your own LGPL compliance for this one file (e.g., providing a way to
relink against a modified `Eigen::SparseCholesky`, or making the combined
work's source available).

## What's actually done about it

Rather than lifting `-DEIGEN_MPL2_ONLY` for the whole fork, the
restriction is undone for just this translation unit:
`libs/tex/CMakeLists.txt` compiles `poisson_blending.cpp` with
`-UEIGEN_MPL2_ONLY` appended, re-enabling `SparseCholesky` there while
every other file in this codebase stays MPL2-only-clean. This keeps the
LGPL surface auditable to exactly one file instead of the whole binary.

If `poisson_blending.cpp`'s use of `SimplicialLDLT`/`AMDOrdering` is ever
removed, delete the `set_source_files_properties(...)` block in
`libs/tex/CMakeLists.txt` along with it — there's no other reason to keep
the per-file exception around.
