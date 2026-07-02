# Engineering Note: Resolving Eigen `SparseCholesky` Compilation Failures in `mvstexturing`

---

## Context & Root Cause
During the compilation of `mvs-texturing` (specifically `poisson_blending.cpp`), the build fails with the error:  
`#error The SparseCholesky module has nothing to offer in MPL2 only mode`

* **The Cause:** The build environment enforces strict licensing via the `-DEIGEN_MPL2_ONLY` preprocessor macro. The legacy vendored version of the Eigen library used in this codebase contains direct matrix solvers (such as `SimplicialLDLT` and its dependency `AMDOrdering`) that are flagged as non-MPL2 compliant (LGPL).
* **The Risk:** While forcing a bypass via `#undef EIGEN_MPL2_ONLY` or commenting out the error allows compilation and retains full performance, doing so breaks strict licensing compliance, which can cause downstream deployment faults or legal issues for production/enterprise users pulling this branch.

---

## Production-Safe Solutions

To maintain strict production compliance while ensuring successful compilation, use one of the following approaches:

### Solution A: Upgrade Vendored Eigen (Recommended)
* **Action:** Replace the embedded Eigen library headers located in `elibs/eigen/` with **Eigen v3.4 (or newer)**.
* **Why it works:** In Eigen 3.4+, the entire `SparseCholesky` module was completely rewritten and natively relicensed under the **MPL2** license. 
* **Impact:** The code will compile out-of-the-box with `-DEIGEN_MPL2_ONLY` active, mathematical outputs remain identical, and it is 100% compliant for enterprise distribution.

### Solution B: Swap to an MPL2-Compliant Iterative Solver
* **Action:** Modify `poisson_blending.cpp` to swap the direct Cholesky solver out for an iterative solver that is inherently MPL2-compliant across all legacy Eigen versions.
* **Code Modification Example:**
  ```cpp
  // Replace the failing direct solver block around line 151:
  // Eigen::SimplicialLDLT<SpMat, Eigen::Lower> solver;
  // solver.compute(A);

  // Substitute with a compliance-safe Iterative Solver:
  #include <Eigen/IterativeLinearSolvers>

  Eigen::ConjugateGradient<SpMat, Eigen::Lower> solver;
  solver.setTolerance(1e-4); // Adjust for desired precision matching
  solver.compute(A);
Impact: Keeps the exact same legacy codebase intact and legally compliant without changing the third-party dependencies, though it may introduce subtle calculation time variances on massive mesh workloads.
