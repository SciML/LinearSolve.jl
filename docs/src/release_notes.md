# Release Notes

## v4.2

  - New `GESVFactorization` algorithm mirroring LAPACK's `gesv` driver: fresh matrices
    factorize and solve in a single `LAPACK.gesv!` call, and repeat solves with only a new
    `b` reuse the cached factors through an allocation-free `LAPACK.getrs!`.
  - Dense `LUFactorization` refactorizations (`cache.A = X` then `solve!`) now reuse the
    cached pivot vector (and, without `alias_A`, the cached factors buffer) on
    Julia >= 1.11, making warm refactorization solves allocation-free.
  - The dense `LUFactorization` pivot-buffer reuse with `alias_A = true` now also covers
    the generic-kernel path (`NoPivot`/`RowNonZero` pivoting and non-BLAS element types),
    on all supported Julia versions.

## v4.0

  - Batched (matrix) right-hand sides are now supported: `solve(LinearProblem(A, B))` with
    `B::AbstractMatrix` computes the equivalent of `A \ B`, factorizing `A` once and
    returning `sol.u` as a `size(A, 2) × size(B, 2)` matrix. This is a breaking change:
    previously a matrix `b` initialized a vector-shaped `u` and generally errored downstream.
    Batched right-hand sides are supported by the factorization-based algorithms; iterative
    (Krylov) methods throw an informative `ArgumentError` for matrix `b`.

## Upcoming Changes

  - `CudaOffloadFactorization` has been split into two algorithms:
    - `CudaOffloadLUFactorization` - Uses LU factorization for better performance
    - `CudaOffloadQRFactorization` - Uses QR factorization for better numerical stability
  - `CudaOffloadFactorization` is now deprecated and will show a warning suggesting to use one of the new algorithms

## v2.0

  - `LinearCache` changed from immutable to mutable. With this, the out of place interfaces like
    `set_A` were deprecated for simply mutating the cache, `cache.A = ...`. This fixes some
    correctness checks and makes the package more robust while improving performance.
  - The default algorithm is now type-stable and does not rely on a dynamic dispatch for the choice.
  - IterativeSolvers.jl and KrylovKit.jl were made into extension packages.
  - Documentation of the solvers has changed to docstrings
