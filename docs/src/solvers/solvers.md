# [Linear System Solvers](@id nonlinearsystemsolvers)

`solve(prob::LinearProlem,alg;kwargs)`

Solves for ``Au=b`` in the problem defined by `prob` using the algorithm
`alg`. If no algorithm is given, a default algorithm will be chosen.

This page is solely focused on the methods for nonlinear systems.

## Recommended Methods

The default algorithm `nothing` is good for choosing an algorithm that will work,
but one may need to change this to receive more performance or precision. If
more precision is necessary, `QRFactorization()` and `SVDFactorization()` are
the best choices, with SVD being the slowest but most precise.

For efficiency, `RFLUFactorization` is the fastest for dense LU-factorizations.
For sparse LU-factorizations, `KLUFactorization` if there is less structure
to the sparsity pattern and `UMFPACKFactorization` if there is more structure.
Pardiso.jl's methods are also known to be very efficient sparse linear solvers.

As sparse matrices get larger, iterative solvers tend to get more efficient than
factorization methods if a lower tolerance of the solution is required.
IterativeSolvers.jl uses a low-rank Q update in its GMRES so it tends to be
faster than Krylov.jl for CPU-based arrays, but it's only compatible with
CPU-based arrays whilc Krylov.jl is more general and will support accelerators
like CUDA.

## Full List of Methods

### RecursiveFactorization.jl

- `RFLUFactorization()`: a fast pure Julia LU-factorization implementation
  using RecursiveFactorization.jl. This is by far the fastest LU-factorization
  implementation, usually outperforming OpenBLAS and MKL, but generally optimized
  only for Base `Array` with `Float32`, `Float64`, `ComplexF32`, and `ComplexF64`.

### Base.LinearAlgebra

These overloads tend to work for many array types, such as `CuArrays` for GPU-accelerated
solving, using the overloads provided by the respective packages.

- `LUFactorization(pivot=LinearAlgebra.RowMaximum())`: Julia's built in `lu`.
  Uses the current BLAS implementation of the user's computer.
- `QRFactorization(pivot=LinearAlgebra.NoPivot(),blocksize=16)`: Julia's built in `qr`.
  Uses the current BLAS implementation of the user's computer.
- `SVDFactorization(full=false,alg=LinearAlgebra.DivideAndConquer())`: Julia's built in `svd`.
  Uses the current BLAS implementation of the user's computer.
- `GenericFactorization(fact_alg)`: Constructs a linear solver from a generic
  factorization algorithm `fact_alg` which complies with the Base.LinearAlgebra
  factorization API.

### SuiteSparse.jl

By default, the SuiteSparse.jl are implemented for efficiency by caching the
symbolic factorization. I.e. if `set_A` is used, it is expected that the new
`A` has the same sparsity pattern as the previous `A`. If this algorithm is to
be used in a context where that assumption does not hold, set `reuse_symbolic=false`.

- `KLUFactorization(;reuse_symbolic=true)`: A fast sparse LU-factorization which
  specializes on sparsity patterns with "less structure".
- `UMFPACKFactorization(;reuse_symbolic=true)`: A fast sparse multithreaded
  LU-factorization which specializes on sparsity patterns that are more
  structured.

### Pardiso.jl

This package is not loaded by default. Thus in order to use this package you
must first `using Pardiso`. The following algorithms are pre-specified:

- `MKLPardisoFactorize(;kwargs...)`: A sparse factorization method.
- `MKLPardisoIterate(;kwargs...)`: A mixed factorization+iterative method.

Those algorithms are defined via:

```julia
MKLPardisoFactorize(;kwargs...) = PardisoJL(;fact_phase=Pardiso.NUM_FACT,
                                             solve_phase=Pardiso.SOLVE_ITERATIVE_REFINE,
                                             kwargs...)
MKLPardisoIterate(;kwargs...) = PardisoJL(;solve_phase=Pardiso.NUM_FACT_SOLVE_REFINE,
                                           kwargs...)
```

The full set of keyword arguments for `PardisoJL` are:                         

```julia
Base.@kwdef struct PardisoJL <: SciMLLinearSolveAlgorithm
    nprocs::Union{Int, Nothing} = nothing
    solver_type::Union{Int, Pardiso.Solver, Nothing} = nothing
    matrix_type::Union{Int, Pardiso.MatrixType, Nothing} = nothing
    fact_phase::Union{Int, Pardiso.Phase, Nothing} = nothing
    solve_phase::Union{Int, Pardiso.Phase, Nothing} = nothing
    release_phase::Union{Int, Nothing} = nothing
    iparm::Union{Vector{Tuple{Int,Int}}, Nothing} = nothing
    dparm::Union{Vector{Tuple{Int,Int}}, Nothing} = nothing
end
```

### IterativeSolvers.jl

- `IterativeSolversJL_CG(args...;kwargs...)`: A generic CG implementation
- `IterativeSolversJL_GMRES(args...;kwargs...)`: A generic GMRES implementation
- `IterativeSolversJL_BICGSTAB(args...;kwargs...)`: A generic BICGSTAB implementation
- `IterativeSolversJL_MINRES(args...;kwargs...)`: A generic MINRES implementation

The general algorithm is:

```julia
IterativeSolversJL(args...;
                   generate_iterator = IterativeSolvers.gmres_iterable!,
                   Pl=nothing, Pr=nothing,
                   gmres_restart=0, kwargs...)
```

### Krylov.jl

- `KrylovJL_CG(args...;kwargs...)`: A generic CG implementation
- `KrylovJL_GMRES(args...;kwargs...)`: A generic GMRES implementation
- `KrylovJL_BICGSTAB(args...;kwargs...)`: A generic BICGSTAB implementation
- `KrylovJL_MINRES(args...;kwargs...)`: A generic MINRES implementation

The general algorithm is:

```julia
KrylovJL(args...; KrylovAlg = Krylov.gmres!,
                  Pl=nothing, Pr=nothing,
                  gmres_restart=0, window=0,
                  kwargs...)
```
