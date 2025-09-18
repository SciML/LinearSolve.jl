# Release Notes

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
