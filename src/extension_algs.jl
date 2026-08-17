# This file only include the algorithm struct to be exported by LinearSolve.jl. The main
# functionality is implemented as package extensions
"""
    PETScAlgorithm(solver_type = :gmres; kwargs...)

A `LinearSolve.jl` algorithm that wraps PETSc's KSP (Krylov Subspace) linear
solvers via [PETSc.jl](https://github.com/JuliaParallel/PETSc.jl).

!!! compat
    Requires `PETSc.jl`, `MPI.jl`, and `SparseMatricesCSR.jl` to be loaded:
    ```julia
    using PETSc, MPI, SparseMatricesCSR
    MPI.Init()
    ```

!!! warning "Serial and MPI-parallel"
    Standard Julia matrices use serial solves via `MPI.COMM_SELF` unless a
    non-`nothing` communicator is supplied. Distributed `PSparseMatrix` and
    `PVector` inputs are handled by the MPI extension when `PETSc` and
    `PartitionedArrays` are loaded.

!!! note "Replicated SparseMatrixCSC with MPI"
    Plain Julia sparse matrices such as `SparseMatrixCSC` can also be solved on a
    multi-rank communicator by passing `comm = MPI.COMM_WORLD` (or another MPI
    communicator). Each rank assembles only its owned row interval into PETSc, PETSc
    solves the distributed system, and the final `sol.u` is gathered back as the full
    Julia vector on every rank.

---

## Positional Arguments

- `solver_type::Symbol` — PETSc KSP solver type.
  Common values: `:gmres` (default), `:cg`, `:bcgs`, `:bicg`, `:preonly`,
  `:richardson`.

---

## Keyword Arguments

| Keyword | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `pc_type` | `Symbol` | `:none` | Preconditioner type: `:jacobi`, `:ilu`, `:lu`, `:gamg`, `:hypre`, … |
| `comm` | `MPI.Comm` | `nothing` | MPI communicator. `nothing` maps to `MPI.COMM_SELF` at solve time. |
| `nullspace` | `Symbol` | `:none` | Null-space strategy: `:none`, `:constant`, or `:custom`. |
| `nullspace_vecs` | `Vector` | `nothing` | Orthonormal null-space basis; required when `nullspace = :custom`. |
| `prec_matrix` | `AbstractMatrix` | `nothing` | Separate matrix used only for building the preconditioner. |
| `initial_guess_nonzero` | `Bool` | `false` | Use the current solution vector as the initial Krylov guess. |
| `transposed` | `Bool` | `false` | Solve the transposed system `Aᵀx = b`. |
| `ksp_options` | `NamedTuple` | `(;)` | Extra PETSc Options Database flags (see table below). |

### Common `ksp_options`

| Option | Description |
| :--- | :--- |
| `ksp_monitor = ""` | Print residual norm each iteration. |
| `ksp_view = ""` | Print solver configuration after setup. |
| `pc_factor_levels = 2` | Fill levels for ILU. |
| `log_view = ""` | PETSc performance logging summary. |

---

## Memory Management

PETSc objects live in C-side memory outside Julia's GC. Call
`cleanup_petsc_cache!` explicitly when finished with a solve to release
resources promptly:

```julia
PETScExt = Base.get_extension(LinearSolve, :LinearSolvePETScExt)

cache = SciMLBase.init(prob, PETScAlgorithm(:gmres))
sol = solve!(cache)
PETScExt.cleanup_petsc_cache!(cache)
```

A GC finalizer is registered as a safety net, but explicit cleanup is
strongly preferred for deterministic, timely resource release.

---

## Example

```julia
using LinearSolve, PETSc, MPI, SparseArrays, SparseMatricesCSR, LinearAlgebra

MPI.Init()
PETScExt = Base.get_extension(LinearSolve, :LinearSolvePETScExt)

n = 100
A = sprand(n, n, 0.1); A = A + A' + 20I
b = rand(n)

cache = SciMLBase.init(
    LinearProblem(A, b),
    PETScAlgorithm(:gmres; pc_type = :ilu, ksp_options = (ksp_monitor = "",))
)
sol = solve!(cache)
println("Residual: ", norm(A * sol.u - b) / norm(b))
PETScExt.cleanup_petsc_cache!(cache)
```

## Distributed SparseMatrixCSC Example

```julia
using LinearSolve, PETSc, MPI, SparseArrays, SparseMatricesCSR, LinearAlgebra

MPI.Init()
PETScExt = Base.get_extension(LinearSolve, :LinearSolvePETScExt)

n = 12
A = spdiagm(-1 => -ones(n - 1), 0 => 4.0 .* ones(n), 1 => -ones(n - 1))
b = ones(n)

cache = SciMLBase.init(
    LinearProblem(A, b),
    PETScAlgorithm(:gmres; comm = MPI.COMM_WORLD);
    abstol = 1.0e-10,
    reltol = 1.0e-10
)
sol = solve!(cache)

# sol.u is the full replicated solution on every rank.
println(norm(A * sol.u - b) / norm(b))
PETScExt.cleanup_petsc_cache!(cache)
```
"""
struct PETScAlgorithm <: SciMLLinearSolveAlgorithm
    solver_type::Symbol
    pc_type::Symbol
    comm::Any             # MPI.Comm, stored as Any to avoid an MPI.jl dependency in LinearSolve
    nullspace::Symbol     # :none | :constant | :custom
    nullspace_vecs::Any   # nothing | Vector of AbstractVectors
    prec_matrix::Any      # nothing | AbstractMatrix
    initial_guess_nonzero::Bool
    transposed::Bool
    ksp_options::NamedTuple

    function PETScAlgorithm(
            solver_type::Symbol = :gmres;
            pc_type::Symbol = :none,
            comm = nothing,
            nullspace::Symbol = :none,
            nullspace_vecs = nothing,
            prec_matrix = nothing,
            initial_guess_nonzero::Bool = false,
            transposed::Bool = false,
            ksp_options::NamedTuple = NamedTuple(),
        )
        Base.get_extension(@__MODULE__, :LinearSolvePETScExt) === nothing && error(
            "PETScAlgorithm requires PETSc, MPI, and SparseMatricesCSR to be loaded: `using PETSc, MPI, SparseMatricesCSR`"
        )
        nullspace ∈ (:none, :constant, :custom) || error(
            "nullspace must be :none, :constant, or :custom (got :$nullspace)"
        )
        nullspace == :custom && nullspace_vecs === nothing && error(
            "nullspace = :custom requires nullspace_vecs to be provided"
        )
        return new(
            solver_type, pc_type, comm,
            nullspace, nullspace_vecs,
            prec_matrix,
            initial_guess_nonzero, transposed,
            ksp_options,
        )
    end
end

# PETSc assembles an AIJ matrix from the entries of `A`.
needs_concrete_A(::PETScAlgorithm) = true

"""
`HYPREAlgorithm(solver; comm = nothing)`

[HYPRE.jl](https://github.com/fredrikekre/HYPRE.jl) is an interface to
[`hypre`](https://computing.llnl.gov/projects/hypre-scalable-linear-solvers-multigrid-methods)
and provide iterative solvers and preconditioners for sparse linear systems. It is mainly
developed for large multi-process distributed problems (using MPI), but can also be used for
single-process problems with Julias standard sparse matrices.

If you need more fine-grained control over the solver/preconditioner options you can
alternatively pass an already created solver to `HYPREAlgorithm` (and to the `Pl` keyword
argument). See HYPRE.jl docs for how to set up solvers with specific options.

!!! note

    Using HYPRE solvers requires Julia version 1.9 or higher, and that the package HYPRE.jl
    is installed.

## Positional Arguments

The single positional argument `solver` has the following choices:

  - `HYPRE.BiCGSTAB`
  - `HYPRE.BoomerAMG`
  - `HYPRE.FlexGMRES`
  - `HYPRE.GMRES`
  - `HYPRE.Hybrid`
  - `HYPRE.ILU`
  - `HYPRE.ParaSails` (as preconditioner only)
  - `HYPRE.PCG`

## Keyword Arguments

  - `comm`: optional MPI communicator used to auto-construct distributed
    `HYPREMatrix` / `HYPREVector` inputs from plain Julia sparse matrices and vectors.

Preconditioners are passed to `solve`/`init` via the `Pl` keyword argument.

## Example

For example, to use `HYPRE.PCG` as the solver, with `HYPRE.BoomerAMG` as the preconditioner,
the algorithm should be defined as follows:

```julia
using HYPRE

HYPRE.Init()
A, b = setup_system(...)
prob = LinearProblem(A, b)
alg = HYPREAlgorithm(HYPRE.PCG)
prec = HYPRE.BoomerAMG
sol = solve(prob, alg; Pl = prec)
```

For automatic distributed construction from a plain Julia sparse matrix on an MPI
communicator, pass the communicator through `comm`:

```julia
using HYPRE, MPI

MPI.Init()
HYPRE.Init()
alg = HYPREAlgorithm(HYPRE.PCG; comm = MPI.COMM_WORLD)
sol = solve(prob, alg)
```
"""
struct HYPREAlgorithm <: SciMLLinearSolveAlgorithm
    solver::Any
    comm::Any
    function HYPREAlgorithm(solver; comm = nothing)
        ext = Base.get_extension(@__MODULE__, :LinearSolveHYPREExt)
        if ext === nothing
            error("HYPREAlgorithm requires that HYPRE is loaded, i.e. `using HYPRE`")
        else
            return new{}(solver, comm)
        end
    end
end

# HYPRE builds its own IJMatrix from the entries of `A`.
needs_concrete_A(::HYPREAlgorithm) = true

"""
    PartitionedSolversAlgorithm(solver = nothing; kwargs...)

[PartitionedSolvers](https://github.com/PartitionedArrays/PartitionedArrays.jl/tree/master/PartitionedSolvers)
provides distributed linear solver building blocks for
[PartitionedArrays.jl](https://github.com/PartitionedArrays/PartitionedArrays.jl).

This algorithm requires `PSparseMatrix` / `PVector` inputs: `init` throws an
`ArgumentError` if `A` is not a `PSparseMatrix`, if `b` is not a `PVector`, or if a
`u0` is given that is not a `PVector`. The integration delegates actual solves to the
local `PartitionedSolvers` solver constructors and caches the resulting solver object
for repeated solves. The default dispatch for `PSparseMatrix` inputs chooses the
CG-backed PartitionedSolvers path, but the integration is solver-agnostic: any
PartitionedSolvers solver constructor (for example `PartitionedSolvers.cg`,
`PartitionedSolvers.jacobi`, or `PartitionedSolvers.amg`) can be passed, and only the
convergence keywords that the chosen solver actually accepts (`iterations`, `abstol`,
`reltol`, `verbose`, and `Pl` when a left preconditioner is set, otherwise
`update_Pl = false`) are forwarded automatically.

!!! note

    Using this solver requires that both PartitionedArrays.jl and PartitionedSolvers.jl
    are loaded, i.e. `using PartitionedArrays, PartitionedSolvers`. The constructor
    errors otherwise.

## Positional Arguments

  - `solver`: optional PartitionedSolvers solver constructor such as
    `PartitionedSolvers.cg`, or an already constructed `PartitionedSolvers.AbstractSolver`.
    Defaults to `nothing`, in which case `PartitionedSolvers.default_solver(problem)` is
    used.

## Keyword Arguments

  - `kwargs...`: forwarded when a solver constructor is called for the problem. They take
    precedence over the auto-derived convergence keywords. They are not used when `solver`
    is `nothing` (the default solver takes no options) or when `solver` is an already
    constructed `AbstractSolver` (which is updated with the new problem via
    `PartitionedSolvers.update` instead).

## Example

```julia
using LinearSolve, PartitionedArrays, PartitionedSolvers

alg = PartitionedSolversAlgorithm(PartitionedSolvers.cg)
sol = solve(prob, alg)
```
"""
struct PartitionedSolversAlgorithm <: SciMLLinearSolveAlgorithm
    solver::Any
    kwargs::NamedTuple
    function PartitionedSolversAlgorithm(solver = nothing; kwargs...)
        ext = Base.get_extension(@__MODULE__, :LinearSolvePartitionedSolversExt)
        if ext === nothing
            error(
                "PartitionedSolversAlgorithm requires PartitionedArrays and PartitionedSolvers to be loaded: `using PartitionedArrays, PartitionedSolvers`"
            )
        else
            return new(solver, NamedTuple(kwargs))
        end
    end
end

# PartitionedSolvers operates on the entries of a `PSparseMatrix`.
needs_concrete_A(::PartitionedSolversAlgorithm) = true

# Debug: About to define CudaOffloadLUFactorization
"""
    CudaOffloadLUFactorization(; throwerror = true, residualsafety = false)

An offloading technique used to GPU-accelerate CPU-based computations using LU factorization.
The dense CPU matrix `A` is copied to a `CuArray`, factored with cuSOLVER's `lu`, and each
solve copies `b` to the GPU, back-solves there, and copies the result back to `cache.u`.
Requires a sufficiently large `A` to overcome the data transfer costs.

## Keyword Arguments

  - `throwerror`: whether to throw an error at construction when the CUDA extension is not
    loaded. Defaults to `true`. Passing `false` is what lets the default solver build the
    algorithm speculatively.
  - `residualsafety`: intended to enable the post-solve residual check described for
    `LUFactorization`. Defaults to `false`. That check lives in the generic factorization
    `solve!`, which the CUDA extension replaces with its own `solve!` for this algorithm,
    so the flag currently has no effect here.

!!! note

    Using this solver requires adding the package CUDA.jl, i.e. `using CUDA`
"""
struct CudaOffloadLUFactorization <: AbstractFactorization
    residualsafety::Bool
    function CudaOffloadLUFactorization(; throwerror = true, residualsafety::Bool = false)
        ext = Base.get_extension(@__MODULE__, :LinearSolveCUDAExt)
        if ext === nothing && throwerror
            error("CudaOffloadLUFactorization requires that CUDA is loaded, i.e. `using CUDA`")
        else
            return new(residualsafety)
        end
    end
end

"""
    CUDAOffload32MixedLUFactorization(; throwerror = true)

A mixed precision GPU-accelerated LU factorization that converts matrices to Float32
(ComplexF32 for complex inputs) before offloading to CUDA GPU for factorization, then
converts back for the solve. This can provide speedups when the reduced precision is
acceptable and memory bandwidth is a bottleneck.

## Keyword Arguments

  - `throwerror`: whether to throw an error at construction when the CUDA extension is not
    loaded. Defaults to `true`.

## Performance Notes
- Converts Float64 matrices to Float32 for GPU factorization
- Can be significantly faster for large matrices where memory bandwidth is limiting
- May have reduced accuracy compared to full precision methods
- Most beneficial when the condition number of the matrix is moderate

!!! note

    Using this solver requires adding the package CUDA.jl, i.e. `using CUDA`
"""
struct CUDAOffload32MixedLUFactorization <: AbstractFactorization
    function CUDAOffload32MixedLUFactorization(; throwerror = true)
        ext = Base.get_extension(@__MODULE__, :LinearSolveCUDAExt)
        if ext === nothing && throwerror
            error("CUDAOffload32MixedLUFactorization requires that CUDA is loaded, i.e. `using CUDA`")
        else
            return new()
        end
    end
end

"""
`CudaOffloadQRFactorization()`

An offloading technique used to GPU-accelerate CPU-based computations using QR factorization.
Requires a sufficiently large `A` to overcome the data transfer costs.

!!! note

    Using this solver requires adding the package CUDA.jl, i.e. `using CUDA`
"""
struct CudaOffloadQRFactorization <: AbstractFactorization
    function CudaOffloadQRFactorization()
        ext = Base.get_extension(@__MODULE__, :LinearSolveCUDAExt)
        if ext === nothing
            error("CudaOffloadQRFactorization requires that CUDA is loaded, i.e. `using CUDA`")
        else
            return new()
        end
    end
end

"""
`CudaOffloadFactorization()`

!!! warning
    This algorithm is deprecated. Use `CudaOffloadLUFactorization` or `CudaOffloadQRFactorization()` instead.

An offloading technique used to GPU-accelerate CPU-based computations.
Requires a sufficiently large `A` to overcome the data transfer costs.

!!! note

    Using this solver requires adding the package CUDA.jl, i.e. `using CUDA`
"""
struct CudaOffloadFactorization <: AbstractFactorization
    function CudaOffloadFactorization()
        Base.depwarn("`CudaOffloadFactorization` is deprecated, use `CudaOffloadLUFactorization` or `CudaOffloadQRFactorization` instead.", :CudaOffloadFactorization)
        ext = Base.get_extension(@__MODULE__, :LinearSolveCUDAExt)
        if ext === nothing
            error("CudaOffloadFactorization requires that CUDA is loaded, i.e. `using CUDA`")
        else
            return new()
        end
    end
end

"""
`AMDGPUOffloadLUFactorization()`

An offloading technique using LU factorization to GPU-accelerate CPU-based computations on AMD GPUs.
Requires a sufficiently large `A` to overcome the data transfer costs.

!!! note

    Using this solver requires adding the package AMDGPU.jl, i.e. `using AMDGPU`
"""
struct AMDGPUOffloadLUFactorization <: LinearSolve.AbstractFactorization
    function AMDGPUOffloadLUFactorization()
        ext = Base.get_extension(@__MODULE__, :LinearSolveAMDGPUExt)
        if ext === nothing
            error("AMDGPUOffloadLUFactorization requires that AMDGPU is loaded, i.e. `using AMDGPU`")
        else
            return new{}()
        end
    end
end

"""
`AMDGPUOffloadQRFactorization()`

An offloading technique using QR factorization to GPU-accelerate CPU-based computations on AMD GPUs.
Requires a sufficiently large `A` to overcome the data transfer costs.

!!! note

    Using this solver requires adding the package AMDGPU.jl, i.e. `using AMDGPU`
"""
struct AMDGPUOffloadQRFactorization <: LinearSolve.AbstractFactorization
    function AMDGPUOffloadQRFactorization()
        ext = Base.get_extension(@__MODULE__, :LinearSolveAMDGPUExt)
        if ext === nothing
            error("AMDGPUOffloadQRFactorization requires that AMDGPU is loaded, i.e. `using AMDGPU`")
        else
            return new{}()
        end
    end
end

## RFLUFactorization

"""
    RFLUFactorization(; pivot = Val(true), thread = Val(true), throwerror = true,
                      residualsafety = false)
    RFLUFactorization(pivot::Val, thread::Val; throwerror = true, residualsafety = false)

A fast pure Julia LU-factorization implementation using RecursiveFactorization.jl.
This is by far the fastest LU-factorization implementation, usually outperforming
OpenBLAS and MKL for smaller matrices (<500x500), but currently optimized only for
Base `Array` with `Float32` or `Float64`. Additional optimization for complex matrices
is in the works.

## Type Parameters
- `P`: Pivoting strategy as `Val{Bool}`. `Val{true}` enables partial pivoting for stability.
- `T`: Threading strategy as `Val{Bool}`. `Val{true}` enables multi-threading for performance.

## Constructor Arguments
- `pivot = Val(true)`: Enable partial pivoting. Set to `Val{false}` to disable for speed
  at the cost of numerical stability.
- `thread = Val(true)`: Enable multi-threading. Set to `Val{false}` for single-threaded
  execution.
- `throwerror = true`: Whether to throw an error if RecursiveFactorization.jl is not loaded.
- `residualsafety = false`: intended to enable the post-solve residual check described
  for `LUFactorization`. That check lives in the generic factorization `solve!`, which
  the RecursiveFactorization extension replaces with its own `solve!` for this
  algorithm, so the flag currently has no effect here.

## Performance Notes
- Fastest for dense matrices with dimensions roughly < 500×500
- Optimized specifically for Float32 and Float64 element types
- Recursive blocking strategy provides excellent cache performance
- Multi-threading can provide significant speedups on multi-core systems

## Requirements
Using this solver requires that RecursiveFactorization.jl is loaded: `using RecursiveFactorization`

## Example
```julia
using RecursiveFactorization
# Fast, stable (with pivoting)
alg1 = RFLUFactorization()
# Fastest (no pivoting), less stable
alg2 = RFLUFactorization(pivot=Val(false))
```
"""
struct RFLUFactorization{P, T} <: AbstractDenseFactorization
    residualsafety::Bool
    function RFLUFactorization(::Val{P}, ::Val{T}; throwerror = true, residualsafety::Bool = false) where {P, T}
        if !userecursivefactorization(nothing)
            throwerror &&
                error("RFLUFactorization requires that RecursiveFactorization.jl is loaded, i.e. `using RecursiveFactorization`")
        end
        return new{P, T}(residualsafety)
    end
end

function RFLUFactorization(; pivot = Val(true), thread = Val(true), throwerror = true, residualsafety::Bool = false)
    return RFLUFactorization(pivot, thread; throwerror, residualsafety)
end

"""
    ButterflyFactorization(; thread = Val(true), throwerror = true)
    ButterflyFactorization(thread::Val; throwerror = true)

A fast pure Julia LU-factorization implementation using RecursiveFactorization.jl.
Instead of pivoting, the matrix is first mixed with RecursiveFactorization.jl's random
butterfly transform and then factored with the pivot-free `RecursiveFactorization.lu!`,
and each solve applies the transform to the right-hand side around a `TriangularSolve`
back-solve. This trades the row swaps of partial pivoting for two cheap dense
multiplies, which can be faster than `RFLUFactorization` on dense square `Array` inputs
while being less robust for matrices that need pivoting for stability. Only square
matrices are supported (an assertion fires on the first factorization otherwise).

## Keyword Arguments

  - `thread`: threading choice as `Val{Bool}`, forwarded to
    `RecursiveFactorization.lu!` and to the triangular back-solves. Defaults to
    `Val(true)`.
  - `throwerror`: whether to throw an error at construction when
    RecursiveFactorization.jl is not loaded. Defaults to `true`.

!!! note

    Using this solver requires that RecursiveFactorization.jl is loaded, i.e.
    `using RecursiveFactorization`.
"""
struct ButterflyFactorization{T} <: AbstractDenseFactorization
    thread::Val{T}
    function ButterflyFactorization(::Val{T}; throwerror = true) where {T}
        if !userecursivefactorization(nothing)
            throwerror &&
                error("ButterflyFactorization requires that RecursiveFactorization.jl is loaded, i.e. `using RecursiveFactorization`")
        end
        return new{T}()
    end
end

function ButterflyFactorization(; thread = Val(true), throwerror = true)
    return ButterflyFactorization(thread; throwerror)
end


# There's no options like pivot here.
# But I'm not sure it makes sense as a GenericFactorization
# since it just uses `LAPACK.getrf!`.
"""
    FastLUFactorization()

A high-performance LU factorization using the FastLapackInterface.jl package.
This provides an optimized interface to LAPACK routines with reduced overhead
compared to the standard LinearAlgebra LAPACK wrappers.

## Features
- Reduced function call overhead compared to standard LAPACK wrappers
- Optimized for performance-critical applications
- Uses partial pivoting (no choice of pivoting method available)
- Suitable for dense matrices where maximum performance is required

## Limitations
- Does not allow customization of pivoting strategy (always uses partial pivoting)
- Requires FastLapackInterface.jl to be loaded
- Limited to dense matrix types supported by LAPACK

## Requirements
Using this solver requires that FastLapackInterface.jl is loaded: `using FastLapackInterface`

## Performance Notes
This factorization is optimized for cases where the overhead of standard LAPACK
function calls becomes significant, typically for moderate-sized dense matrices
or when performing many factorizations.

## Example
```julia
using FastLapackInterface
alg = FastLUFactorization()
sol = solve(prob, alg)
```
"""
struct FastLUFactorization <: AbstractDenseFactorization end

"""
    FastQRFactorization()
    FastQRFactorization(pivot, blocksize)

A high-performance QR factorization using the FastLapackInterface.jl package.
This provides an optimized interface to LAPACK QR routines with reduced overhead
compared to the standard LinearAlgebra LAPACK wrappers. The zero-argument constructor
is `FastQRFactorization(NoPivot(), 36)`; there is no keyword constructor, so both
fields are set positionally.

## Type Parameters
- `P`: The type of pivoting strategy used

## Positional Arguments
- `pivot`: Pivoting strategy. `NoPivot()` (the default) uses the blocked
  unpivoted `geqrt!` path; `ColumnNorm()` selects the column-pivoted `geqp3!` path.
  These are the only two pivot types the extension implements.
- `blocksize`: Block size for the blocked (unpivoted) QR algorithm, passed to
  FastLapackInterface's `QRWYWs` workspace. Defaults to `36`. It is not used by the
  column-pivoted path.

## Features
- Reduced function call overhead compared to standard LAPACK wrappers
- Configurable block size for optimal performance
- Suitable for dense matrices, especially overdetermined systems

## Performance Notes
The block size can be tuned for optimal performance depending on matrix size and architecture.
The default value of 36 is generally good for most cases, but experimentation may be beneficial
for specific applications.

## Requirements
Using this solver requires that FastLapackInterface.jl is loaded: `using FastLapackInterface`

## Example
```julia
using FastLapackInterface, LinearAlgebra
# QR without pivoting, block size 36
alg1 = FastQRFactorization()
# QR with column pivoting
alg2 = FastQRFactorization(ColumnNorm(), 36)
# Custom block size
alg3 = FastQRFactorization(NoPivot(), 64)
```
"""
struct FastQRFactorization{P} <: AbstractDenseFactorization
    pivot::P
    blocksize::Int
end

# is 36 or 16 better here? LinearAlgebra and FastLapackInterface use 36,
# but QRFactorization uses 16.
FastQRFactorization() = FastQRFactorization(NoPivot(), 36)

"""
```julia
MKLPardisoFactorize(; nprocs::Union{Int, Nothing} = nothing,
    matrix_type = nothing,
    cache_analysis = false,
    iparm::Union{Vector{Tuple{Int, Int}}, Nothing} = nothing,
    dparm::Union{Vector{Tuple{Int, Int}}, Nothing} = nothing)
```

A sparse direct (LU) factorization method using MKL Pardiso, i.e.
`PardisoJL(; vendor = :MKL, solver_type = 0, kwargs...)`. It solves square sparse
systems (`SparseMatrixCSC`, real or complex element types) with a multithreaded
supernodal LU and is a good choice for large sparse systems on Intel hardware where
MKL is available. Use `MKLPardisoIterate` when the same sparsity pattern is factored
repeatedly and the LU-preconditioned iteration is likely to converge without a
refactorization.

!!! note

    Using this solver requires adding the package Pardiso.jl, i.e. `using Pardiso`

## Keyword Arguments

  - `nprocs`: number of threads, passed to `Pardiso.set_nprocs!`. Defaults to `nothing`
    (Pardiso's default).
  - `matrix_type`: Pardiso matrix type (`Pardiso.MatrixType` or its integer code),
    overriding the automatic choice of `Pardiso.REAL_NONSYM` for real and
    `Pardiso.COMPLEX_NONSYM` for complex element types. Defaults to `nothing`.
  - `cache_analysis`: when `true`, disables Pardiso's scaling and matching defaults
    (`iparm[11] = iparm[13] = 0`), runs the analysis phase once at `init`, and reuses it
    for all further factorizations with this solver. Defaults to `false`.
  - `iparm`: vector of `(index, value)` tuples applied via `Pardiso.set_iparm!`.
    Defaults to `nothing`.
  - `dparm`: vector of `(index, value)` tuples applied via `Pardiso.set_dparm!`.
    Defaults to `nothing`.

The defaults let the solver determine everything from the input types; these keywords
are only for overriding that handling and should not be required by most users. See
`PardisoJL` for the full description of each keyword and the Pardiso.jl documentation
for the meaning of the individual `iparm`/`dparm` entries.
"""
MKLPardisoFactorize(; kwargs...) = PardisoJL(; vendor = :MKL, solver_type = 0, kwargs...)

"""
```julia
MKLPardisoIterate(; nprocs::Union{Int, Nothing} = nothing,
    matrix_type = nothing,
    cache_analysis = false,
    iparm::Union{Vector{Tuple{Int, Int}}, Nothing} = nothing,
    dparm::Union{Vector{Tuple{Int, Int}}, Nothing} = nothing)
```

A mixed factorization+iterative method using MKL Pardiso, i.e.
`PardisoJL(; vendor = :MKL, solver_type = 1, kwargs...)`. Pardiso computes an LU
factorization for the first system and then reuses those exact factors as the
preconditioner of a Krylov (CGS) iteration for the following solves, falling back to a
fresh numerical factorization when the iteration does not converge. `iparm[4]` is set
from the `reltol` given to `init`/`solve` to control the iteration's stopping
tolerance. This is worthwhile when the same sparsity pattern is solved many times with
slowly changing values (for example inside a nonlinear or time-stepping loop) and each
new factorization is expensive; use `MKLPardisoFactorize` for a plain direct solve.
Square sparse systems (`SparseMatrixCSC`, real or complex) are supported.

!!! note

    Using this solver requires adding the package Pardiso.jl, i.e. `using Pardiso`

## Keyword Arguments

  - `nprocs`: number of threads, passed to `Pardiso.set_nprocs!`. Defaults to `nothing`
    (Pardiso's default).
  - `matrix_type`: Pardiso matrix type (`Pardiso.MatrixType` or its integer code),
    overriding the automatic choice of `Pardiso.REAL_NONSYM` for real and
    `Pardiso.COMPLEX_NONSYM` for complex element types. Defaults to `nothing`.
  - `cache_analysis`: when `true`, disables Pardiso's scaling and matching defaults
    (`iparm[11] = iparm[13] = 0`), runs the analysis phase once at `init`, and reuses it
    for all further factorizations with this solver. Defaults to `false`.
  - `iparm`: vector of `(index, value)` tuples applied via `Pardiso.set_iparm!`.
    Defaults to `nothing`.
  - `dparm`: vector of `(index, value)` tuples applied via `Pardiso.set_dparm!`.
    Defaults to `nothing`.

The defaults let the solver determine everything from the input types; these keywords
are only for overriding that handling and should not be required by most users. See
`PardisoJL` for the full description of each keyword and the Pardiso.jl documentation
for the meaning of the individual `iparm`/`dparm` entries.
"""
MKLPardisoIterate(; kwargs...) = PardisoJL(; vendor = :MKL, solver_type = 1, kwargs...)

"""
```julia
PanuaPardisoFactorize(; nprocs::Union{Int, Nothing} = nothing,
    matrix_type = nothing,
    cache_analysis = false,
    iparm::Union{Vector{Tuple{Int, Int}}, Nothing} = nothing,
    dparm::Union{Vector{Tuple{Int, Int}}, Nothing} = nothing)
```

A sparse direct (LU) factorization method using Panua Pardiso (formerly
pardiso-project.org), i.e. `PardisoJL(; vendor = :Panua, solver_type = 0, kwargs...)`.
It solves square sparse systems (`SparseMatrixCSC`, real or complex element types) with
a multithreaded supernodal LU and is the choice when a Panua Pardiso license is
available, including on platforms where MKL Pardiso is not. Use `PanuaPardisoIterate` when
the same sparsity pattern is factored repeatedly and the LU-preconditioned iteration is
likely to converge without a refactorization.

!!! note

    Using this solver requires adding the package Pardiso.jl, i.e. `using Pardiso`

## Keyword Arguments

  - `nprocs`: number of threads. Defaults to `nothing` (Pardiso's default). The
    extension currently applies this only for the MKL vendor, so it has no effect here.
  - `matrix_type`: Pardiso matrix type (`Pardiso.MatrixType` or its integer code),
    overriding the automatic choice of `Pardiso.REAL_NONSYM` for real and
    `Pardiso.COMPLEX_NONSYM` for complex element types. Defaults to `nothing`.
  - `cache_analysis`: when `true`, disables Pardiso's scaling and matching defaults
    (`iparm[11] = iparm[13] = 0`), runs the analysis phase once at `init`, and reuses it
    for all further factorizations with this solver. Defaults to `false`.
  - `iparm`: vector of `(index, value)` tuples applied via `Pardiso.set_iparm!`.
    Defaults to `nothing`.
  - `dparm`: vector of `(index, value)` tuples applied via `Pardiso.set_dparm!`.
    Defaults to `nothing`.

The defaults let the solver determine everything from the input types; these keywords
are only for overriding that handling and should not be required by most users. See
`PardisoJL` for the full description of each keyword and the Pardiso.jl documentation
for the meaning of the individual `iparm`/`dparm` entries.
"""
PanuaPardisoFactorize(; kwargs...) = PardisoJL(;
    vendor = :Panua, solver_type = 0, kwargs...
)

"""
```julia
PanuaPardisoIterate(; nprocs::Union{Int, Nothing} = nothing,
    matrix_type = nothing,
    cache_analysis = false,
    iparm::Union{Vector{Tuple{Int, Int}}, Nothing} = nothing,
    dparm::Union{Vector{Tuple{Int, Int}}, Nothing} = nothing)
```

A mixed factorization+iterative method using Panua Pardiso, i.e.
`PardisoJL(; vendor = :Panua, solver_type = 1, kwargs...)`. `Pardiso.set_solver!` selects
Panua's iterative solver: an LU factorization is computed for the first system and then
reused as the preconditioner of a Krylov iteration for the following solves, with a
fresh numerical factorization when the iteration does not converge. `iparm[4]` is set
from the `reltol` given to `init`/`solve` to control the iteration's stopping tolerance.
This is worthwhile when the same sparsity pattern is solved many times with slowly
changing values and each new factorization is expensive; use `PanuaPardisoFactorize`
for a plain direct solve. Square sparse systems (`SparseMatrixCSC`, real or complex)
are supported.

!!! note

    Using this solver requires adding the package Pardiso.jl, i.e. `using Pardiso`

## Keyword Arguments

  - `nprocs`: number of threads. Defaults to `nothing` (Pardiso's default). The
    extension currently applies this only for the MKL vendor, so it has no effect here.
  - `matrix_type`: Pardiso matrix type (`Pardiso.MatrixType` or its integer code),
    overriding the automatic choice of `Pardiso.REAL_NONSYM` for real and
    `Pardiso.COMPLEX_NONSYM` for complex element types. Defaults to `nothing`.
  - `cache_analysis`: when `true`, disables Pardiso's scaling and matching defaults
    (`iparm[11] = iparm[13] = 0`), runs the analysis phase once at `init`, and reuses it
    for all further factorizations with this solver. Defaults to `false`.
  - `iparm`: vector of `(index, value)` tuples applied via `Pardiso.set_iparm!`.
    Defaults to `nothing`.
  - `dparm`: vector of `(index, value)` tuples applied via `Pardiso.set_dparm!`.
    Defaults to `nothing`.

The defaults let the solver determine everything from the input types; these keywords
are only for overriding that handling and should not be required by most users. See
`PardisoJL` for the full description of each keyword and the Pardiso.jl documentation
for the meaning of the individual `iparm`/`dparm` entries.
"""
PanuaPardisoIterate(; kwargs...) = PardisoJL(; vendor = :Panua, solver_type = 1, kwargs...)

"""
```julia
PardisoJL(; nprocs::Union{Int, Nothing} = nothing,
    solver_type = nothing,
    matrix_type = nothing,
    cache_analysis = false,
    iparm::Union{Vector{Tuple{Int, Int}}, Nothing} = nothing,
    dparm::Union{Vector{Tuple{Int, Int}}, Nothing} = nothing,
    vendor::Union{Symbol, Nothing} = nothing
)
```

A generic sparse direct solver using Pardiso through Pardiso.jl. It supports square
sparse systems (`SparseMatrixCSC`, real or complex element types) and both the Panua
and MKL Pardiso libraries. The convenience constructors `MKLPardisoFactorize`,
`MKLPardisoIterate`, `PanuaPardisoFactorize` and `PanuaPardisoIterate` fix `vendor` and
`solver_type` and are what most users should use; `PardisoJL` itself is for choosing
those settings by hand. `solver_type` is optional: when it is left as `nothing`,
Pardiso's default (direct) solver is used.

!!! note

    Using this solver requires adding the package Pardiso.jl, i.e. `using Pardiso`

## Keyword Arguments

  - `vendor`: `:Panua` for Panua Pardiso (formerly pardiso-project.org) or `:MKL` for
    MKL Pardiso. Defaults to `nothing`, which selects Panua Pardiso when it is available
    and MKL Pardiso otherwise.
  - `solver_type`: `0` for the sparse direct (LU) solver, `1` for the LU-preconditioned
    Krylov iteration (Pardiso factors the first system and reuses those factors as a
    preconditioner for the following solves, refactoring when the iteration does not
    converge). A `Pardiso.Solver` value is also accepted. Defaults to `nothing`, which
    keeps Pardiso's default. For `vendor = :Panua` the value is passed to
    `Pardiso.set_solver!`; for `vendor = :MKL` it is not passed to Pardiso, but
    `solver_type = 1` still sets `iparm[4]` from `reltol` for either vendor, which is
    what enables MKL's CGS iteration.
  - `nprocs`: number of threads, passed to `Pardiso.set_nprocs!`. Defaults to `nothing`
    (Pardiso's default). The extension currently applies this only for `vendor = :MKL`.
  - `matrix_type`: Pardiso matrix type (`Pardiso.MatrixType` or its integer code),
    passed to `Pardiso.set_matrixtype!`. Defaults to `nothing`, which selects
    `Pardiso.REAL_NONSYM` for real and `Pardiso.COMPLEX_NONSYM` for complex element
    types.
  - `cache_analysis`: when `true`, disables Pardiso's scaling and matching defaults
    (`iparm[11] = iparm[13] = 0`), runs the analysis phase once at `init`, and reuses it
    for every later factorization, so only the numerical factorization is repeated when
    `A` changes with the same sparsity pattern. Defaults to `false`, in which case
    analysis and numerical factorization are redone together on each fresh `A`.
  - `iparm`: vector of `(index, value)` tuples, each applied as
    `Pardiso.set_iparm!(solver, index, value)` after the settings above, so they
    override them. Defaults to `nothing`. `iparm[12]` is set by the extension to
    account for the CSC storage and should not be overridden.
  - `dparm`: vector of `(index, value)` tuples, each applied as
    `Pardiso.set_dparm!(solver, index, value)`. Defaults to `nothing`. Note that the
    field is typed `Vector{Tuple{Int, Int}}`, so only integer values can be passed.

The defaults let the solver determine everything from the input types; these keywords
are only for overriding that handling and should not be required by most users. See the
Pardiso.jl documentation for the meaning of the individual `iparm`/`dparm` entries.
"""
struct PardisoJL{T1, T2} <: AbstractSparseFactorization
    nprocs::Union{Int, Nothing}
    solver_type::T1
    matrix_type::T2
    cache_analysis::Bool
    iparm::Union{Vector{Tuple{Int, Int}}, Nothing}
    dparm::Union{Vector{Tuple{Int, Int}}, Nothing}
    vendor::Union{Symbol, Nothing}

    function PardisoJL(;
            nprocs::Union{Int, Nothing} = nothing,
            solver_type = nothing,
            matrix_type = nothing,
            cache_analysis = false,
            iparm::Union{Vector{Tuple{Int, Int}}, Nothing} = nothing,
            dparm::Union{Vector{Tuple{Int, Int}}, Nothing} = nothing,
            vendor::Union{Symbol, Nothing} = nothing
        )
        ext = Base.get_extension(@__MODULE__, :LinearSolvePardisoExt)
        if ext === nothing
            error("PardisoJL requires that Pardiso is loaded, i.e. `using Pardiso`")
        else
            T1 = typeof(solver_type)
            T2 = typeof(matrix_type)
            @assert T1 <: Union{Int, Nothing, ext.Pardiso.Solver}
            @assert T2 <: Union{Int, Nothing, ext.Pardiso.MatrixType}
            return new{T1, T2}(
                nprocs, solver_type, matrix_type, cache_analysis, iparm, dparm, vendor
            )
        end
    end
end

"""
```julia
KrylovKitJL(args...; KrylovAlg = KrylovKit.GMRES, gmres_restart = 0,
    precs = DEFAULT_PRECS, kwargs...)
```

A generic iterative solver wrapping `KrylovKit.linsolve` from KrylovKit.jl. Each solve
calls `KrylovKit.linsolve(A, b, u; atol = abstol, rtol = reltol, maxiter = maxiters,
krylovdim, verbosity, kwargs...)`, so `A` may be any matrix or operator supporting
`mul!`. KrylovKit selects the actual method from its `issymmetric`, `ishermitian` and
`isposdef` keywords (checked automatically for an `AbstractMatrix`, `false` by default
for other operators): CG for Hermitian positive definite input, GMRES otherwise. Use
`KrylovKitJL_CG` and `KrylovKitJL_GMRES` for the common cases; use `KrylovJL_GMRES`
instead when preconditioning is needed, since KrylovKit has no preconditioner support.

!!! note

    Using this solver requires adding the package KrylovKit.jl, i.e. `using KrylovKit`

## Positional Arguments

  - `args...`: stored on the algorithm; not currently used by the solve.

## Keyword Arguments

  - `KrylovAlg`: the KrylovKit algorithm type, `KrylovKit.GMRES` or `KrylovKit.CG`.
    Defaults to `KrylovKit.GMRES`. It is stored on the algorithm but not passed to
    `KrylovKit.linsolve`; the method is chosen by KrylovKit as described above
    (`KrylovKitJL_CG` requests CG by forcing `isposdef = true`).
  - `gmres_restart`: the GMRES restart length, passed to KrylovKit as `krylovdim`.
    `0` means `min(20, size(A, 1))`. Defaults to `0`. Note that KrylovKit's `maxiter`
    (set from `maxiters`) counts restart cycles for GMRES, and the CG iteration cap is
    `krylovdim * maxiter`.
  - `precs`: a preconditioner builder `(A, p) -> (Pl, Pr)`, called at `init` like for
    `KrylovJL`. Defaults to `DEFAULT_PRECS` (identity). KrylovKit ignores
    preconditioners: any non-identity `Pl`/`Pr`, whether from `precs` or from
    `init`/`solve`, only produces a one-time warning and is otherwise dropped.
  - `kwargs...`: any remaining keywords are forwarded to `KrylovKit.linsolve` on every
    solve (for example `issymmetric`, `ishermitian`, `isposdef`, `orth`), on top of the
    `atol`/`rtol`/`maxiter`/`krylovdim`/`verbosity` set by the cache.
"""
struct KrylovKitJL{F, I, P, A, K} <: LinearSolve.AbstractKrylovSubspaceMethod
    KrylovAlg::F
    gmres_restart::I
    precs::P
    args::A
    kwargs::K
end

"""
```julia
KrylovKitJL_CG(args...; kwargs...)
```

A CG implementation for Hermitian (real symmetric) positive definite linear systems
via KrylovKit.jl. It is `KrylovKitJL(args...; KrylovAlg = KrylovKit.CG, kwargs...,
isposdef = true)`, so `isposdef = true` is always forwarded to `KrylovKit.linsolve`;
KrylovKit then runs CG when it also knows the operator is Hermitian, which is detected
automatically for an `AbstractMatrix` but must be stated with `ishermitian = true` (or
`issymmetric = true` for a real problem) for a function-like operator, otherwise
KrylovKit falls back to GMRES. Keyword arguments (`gmres_restart`, `precs`, and any `KrylovKit.linsolve`
keywords) are those of `KrylovKitJL`. There are no `Pl`/`Pr` keywords: KrylovKit does
not support preconditioners, and any set through `precs` or `init`/`solve` are ignored
with a warning.

!!! note

    Using this solver requires adding the package KrylovKit.jl, i.e. `using KrylovKit`
"""
function KrylovKitJL_CG end

"""
```julia
KrylovKitJL_GMRES(args...; gmres_restart = 0, kwargs...)
```

A GMRES implementation for general (square, possibly non-symmetric) linear systems via
KrylovKit.jl. It is `KrylovKitJL(args...; KrylovAlg = KrylovKit.GMRES, kwargs...)`.
Keyword arguments (`gmres_restart`, `precs`, and any `KrylovKit.linsolve` keywords)
are those of `KrylovKitJL`; in particular `gmres_restart` becomes KrylovKit's
`krylovdim`, with `0` meaning `min(20, size(A, 1))`. There are no `Pl`/`Pr` keywords:
KrylovKit does not support preconditioners, and any set through `precs` or
`init`/`solve` are ignored with a warning. Use `KrylovJL_GMRES` when preconditioning
is needed.

!!! note

    Using this solver requires adding the package KrylovKit.jl, i.e. `using KrylovKit`
"""
function KrylovKitJL_GMRES end

"""
```julia
ConjugateGradientsJL(; solver = :cg, precs = nothing, kwargs...)
```

A wrapper over [ConjugateGradients.jl](https://github.com/mcovalt/ConjugateGradients.jl),
which provides CG and BiCGStab.

`solver` selects `:cg` or `:bicgstab`; the convenience constructors
[`ConjugateGradientsJL_CG`](@ref) and [`ConjugateGradientsJL_BICGSTAB`](@ref) set it
for you. Remaining keywords are forwarded to the underlying solver, so
`bicgstab`'s `tolRho` can be passed through.

ConjugateGradients.jl solves for real element types in a plain `Vector` only, so a
complex or otherwise-typed problem is rejected at `init` rather than partway
through a solve.

## Keyword Arguments

  - `solver`: `:cg` (for symmetric positive definite systems) or `:bicgstab` (for
    general square systems). Defaults to `:cg`.
  - `precs`: a preconditioner builder, a function `(A, p) -> (Pl, Pr)`, called at
    `init` and again in `solve!` whenever `A` or `p` has changed; its result becomes
    `cache.Pl`/`cache.Pr`. `nothing` means no preconditioning unless `Pl` is given to
    `init`/`solve`. Defaults to `nothing`. Only a left preconditioner is supported: `Pl`
    is applied as `ldiv!(z, Pl, r)`, and `solve!` throws an `ArgumentError` if a
    non-identity `Pr` is present.
  - `kwargs...`: forwarded to `ConjugateGradients.cg!`/`bicgstab!` on every solve.

Only `reltol` and `maxiters` from the cache are used: they are passed as
ConjugateGradients.jl's `tol` and `maxIter`. `abstol` is not used.

!!! note

    Using this solver requires adding the package ConjugateGradients.jl, i.e.
    `using ConjugateGradients`
"""
struct ConjugateGradientsJL{P, K} <: LinearSolve.AbstractKrylovSubspaceMethod
    solver::Symbol
    precs::P
    kwargs::K
end

"""
```julia
ConjugateGradientsJL_CG(; precs = nothing, kwargs...)
```

A wrapper over the ConjugateGradients.jl CG. See [`ConjugateGradientsJL`](@ref).

!!! note

    Using this solver requires adding the package ConjugateGradients.jl, i.e.
    `using ConjugateGradients`
"""
function ConjugateGradientsJL_CG end

"""
```julia
ConjugateGradientsJL_BICGSTAB(; precs = nothing, kwargs...)
```

A wrapper over the ConjugateGradients.jl BiCGStab. See [`ConjugateGradientsJL`](@ref).

!!! note

    Using this solver requires adding the package ConjugateGradients.jl, i.e.
    `using ConjugateGradients`
"""
function ConjugateGradientsJL_BICGSTAB end

"""
```julia
IterativeSolversJL(args...;
    generate_iterator = IterativeSolvers.gmres_iterable!,
    gmres_restart = 0, precs = DEFAULT_PRECS, kwargs...)
```

A generic wrapper over the IterativeSolvers.jl solvers. The chosen iterator constructor
is called with the cache's `u`, `A`, `b`, tolerances and iteration cap on every fresh
`A` (and, for every iterator other than `gmres_iterable!`, on every solve), and each
`solve!` then steps through the resulting iterable. The convenience
constructors `IterativeSolversJL_CG`, `IterativeSolversJL_GMRES`,
`IterativeSolversJL_IDRS`, `IterativeSolversJL_BICGSTAB` and `IterativeSolversJL_MINRES`
select `generate_iterator` for you.

!!! note

    Using this solver requires adding the package IterativeSolvers.jl, i.e. `using IterativeSolvers`

## Positional Arguments

  - `args...`: passed positionally to the iterator constructor after `u, A, b` for
    `bicgstabl_iterator!` (where the first one is the BiCGStab(l) `l`) and for
    `minres_iterable!` (which takes none). They are not used by the CG, GMRES and
    IDR(s) iterators.

## Keyword Arguments

  - `generate_iterator`: the IterativeSolvers.jl iterator constructor to use, one of
    `IterativeSolvers.cg_iterator!`, `gmres_iterable!`, `idrs_iterable!`,
    `bicgstabl_iterator!` or `minres_iterable!`. Defaults to
    `IterativeSolvers.gmres_iterable!`.
  - `gmres_restart`: the GMRES restart length, passed as `restart` to
    `gmres_iterable!`. `0` means `min(20, size(A, 1))`. Defaults to `0`. Ignored by
    the other iterators.
  - `precs`: a preconditioner builder `(A, p) -> (Pl, Pr)`, called at `init` like for
    `KrylovJL`; the resulting `cache.Pl`/`cache.Pr` are handed to the iterator
    constructor. Defaults to `DEFAULT_PRECS` (identity). Explicit `Pl`/`Pr` are given
    to `init`/`solve`, not to this constructor: a `Pl` or `Pr` keyword passed here would
    be forwarded to the iterator constructor as-is. Which preconditioners each iterator
    supports is listed on the convenience constructors.
  - `kwargs...`: any remaining keywords are forwarded to the iterator constructor on top
    of `abstol`, `reltol` and `maxiter` from the cache (for example `initially_zero`,
    `orth_meth`, `skew_hermitian`). `maxiters` is accepted here as an alias for
    IterativeSolvers' `maxiter` and overrides the cache's `maxiters`. `idrs_s` is read
    by the IDR(s) path.
"""
struct IterativeSolversJL{F, I, P, A, K} <: LinearSolve.AbstractKrylovSubspaceMethod
    generate_iterator::F
    gmres_restart::I
    precs::P
    args::A
    kwargs::K
end

"""
```julia
IterativeSolversJL_CG(args...; kwargs...)
```

A wrapper over the IterativeSolvers.jl CG (`IterativeSolvers.cg_iterator!`) for
symmetric positive definite systems. It is
`IterativeSolversJL(args...; generate_iterator = IterativeSolvers.cg_iterator!, kwargs...)`,
so the keyword arguments (`precs` and any `cg_iterator!` keywords such as
`initially_zero`) are those of `IterativeSolversJL`; `args...` are not used. There are
no `Pl`/`Pr` constructor keywords: preconditioners come from `precs` or from
`init`/`solve`, and passing a `Pl` keyword here would reach `cg_iterator!` as an
unknown keyword. Only left preconditioning is supported; a non-identity `Pr` is dropped
with a `no_right_preconditioning` message.

!!! note

    Using this solver requires adding the package IterativeSolvers.jl, i.e. `using IterativeSolvers`
"""
function IterativeSolversJL_CG end

"""
```julia
IterativeSolversJL_GMRES(args...; gmres_restart = 0, kwargs...)
```

A wrapper over the IterativeSolvers.jl GMRES (`IterativeSolvers.gmres_iterable!`) for
general square systems. It is
`IterativeSolversJL(args...; generate_iterator = IterativeSolvers.gmres_iterable!, kwargs...)`,
so the keyword arguments (`gmres_restart`, `precs` and any `gmres_iterable!` keywords
such as `orth_meth`) are those of `IterativeSolversJL`; `args...` are not used.
`gmres_restart` is the restart length, with `0` meaning `min(20, size(A, 1))`. Both left
and right preconditioning are supported; `Pl`/`Pr` are taken from `precs` or from
`init`/`solve`, not from constructor keywords.

!!! note

    Using this solver requires adding the package IterativeSolvers.jl, i.e. `using IterativeSolvers`
"""
function IterativeSolversJL_GMRES end

"""
```julia
IterativeSolversJL_IDRS(args...; idrs_s = 4, kwargs...)
```

A wrapper over the IterativeSolvers.jl IDR(s) (`IterativeSolvers.idrs_iterable!`) for
general square systems. It is
`IterativeSolversJL(args...; generate_iterator = IterativeSolvers.idrs_iterable!, kwargs...)`,
so the keyword arguments are those of `IterativeSolversJL`; `args...` are not used.
`idrs_s` is the dimension of the shadow space `s` (larger values typically converge in
fewer iterations at more work per iteration). Defaults to `4`. Remaining keywords are
forwarded to `idrs_iterable!`, which only accepts `smoothing` and `verbose`. Only left
preconditioning is supported: `Pl` comes from `precs` or from `init`/`solve` (not from
a constructor keyword), and a right preconditioner is dropped.

!!! note

    Using this solver requires adding the package IterativeSolvers.jl, i.e. `using IterativeSolvers`
"""
function IterativeSolversJL_IDRS end

"""
```julia
IterativeSolversJL_BICGSTAB(args...; kwargs...)
```

A wrapper over the IterativeSolvers.jl BiCGStab(l) (`IterativeSolvers.bicgstabl_iterator!`)
for general square systems. It is
`IterativeSolversJL(args...; generate_iterator = IterativeSolvers.bicgstabl_iterator!, kwargs...)`.
The first positional argument, if given, is the BiCGStab(l) parameter `l`
(IterativeSolvers' default is `2`). The keyword arguments are those of
`IterativeSolversJL`; `maxiters` is mapped to `max_mv_products = 2 * maxiters`, and
remaining keywords (for example `initial_zero`) are forwarded to `bicgstabl_iterator!`.
Only left preconditioning is supported: `Pl` comes from `precs` or from `init`/`solve`
(not from a constructor keyword), and a right preconditioner is dropped.

!!! note

    Using this solver requires adding the package IterativeSolvers.jl, i.e. `using IterativeSolvers`
"""
function IterativeSolversJL_BICGSTAB end

"""
```julia
IterativeSolversJL_MINRES(args...; kwargs...)
```

A wrapper over the IterativeSolvers.jl MINRES (`IterativeSolvers.minres_iterable!`) for
symmetric (Hermitian) indefinite systems. It is
`IterativeSolversJL(args...; generate_iterator = IterativeSolvers.minres_iterable!, kwargs...)`.
`args...` are passed positionally to `minres_iterable!`, which takes none, so leave
them empty. The keyword arguments are those of `IterativeSolversJL`; remaining keywords
(for example `skew_hermitian`, `initially_zero`) are forwarded to `minres_iterable!`.
This iterator accepts no preconditioner at all: any `Pl`/`Pr`, whether from `precs` or
from `init`/`solve`, is silently ignored.

!!! note

    Using this solver requires adding the package IterativeSolvers.jl, i.e. `using IterativeSolvers`
"""
function IterativeSolversJL_MINRES end

"""
```julia
GinkgoJL(args...; KrylovAlg = :gmres, executor = :omp, kwargs...)
```

A generic wrapper over [Ginkgo.jl](https://github.com/youwuyou/Ginkgo.jl) iterative solvers.
Ginkgo is a high-performance numerical linear algebra library that supports multiple backends
including OpenMP, CUDA, HIP, and SYCL, making it suitable for both CPU and GPU computation.

!!! note

    Using this solver requires adding the package Ginkgo.jl, i.e. `using Ginkgo`

## Positional Arguments

  - `args...`: stored on the algorithm; not currently forwarded to Ginkgo.

## Keyword Arguments

  - `KrylovAlg`: The Ginkgo solver to use. Supported values:
    - `:gmres` (default): GMRES, for general non-symmetric systems
      (not yet exposed by Ginkgo.jl v1, so this errors at solve time; use
      `GinkgoJL_CG()` in the meantime)
    - `:cg`: Conjugate Gradient, for symmetric positive definite systems only
  - `executor`: The Ginkgo backend executor, passed to `Ginkgo.create`. Options:
    - `:omp` (default): OpenMP CPU executor
    - `:cuda`: NVIDIA GPU executor
    - `:reference`: Reference (single-threaded) executor
  - `kwargs...`: stored on the algorithm; not currently forwarded to Ginkgo, so extra
    solver options passed here have no effect.

Each solve rebuilds Ginkgo's CSR matrix and dense vectors from the cache and calls the
Ginkgo solver with `maxiters` and `reltol` from the cache. `abstol` and any `Pl`/`Pr`
preconditioners are not used.

!!! warning

    Ginkgo.jl currently only supports `Float32` element types with `Int32` indices for sparse
    matrices. The input matrix and vectors will be converted to `Float32` automatically.

## Example

```julia
using LinearSolve, Ginkgo, SparseArrays
A = sprand(Float32, 100, 100, 0.1)
A = A'A + 30I  # make symmetric positive definite
b = rand(Float32, 100)
prob = LinearProblem(A, b)
sol = solve(prob, GinkgoJL_CG())
```
"""
struct GinkgoJL{F, E, A, K} <: LinearSolve.AbstractKrylovSubspaceMethod
    KrylovAlg::F
    executor::E
    args::A
    kwargs::K
end

# Unlike the other Krylov wrappers, Ginkgo copies `A` into its own device-side
# sparse format instead of applying it as an operator.
needs_concrete_A(::GinkgoJL) = true

"""
```julia
GinkgoJL_CG(args...; executor = :omp, kwargs...)
```

A CG solver via Ginkgo.jl for symmetric positive definite systems. It is
`GinkgoJL(args...; KrylovAlg = :cg, executor, kwargs...)`; see `GinkgoJL` for the full
description.

## Keyword Arguments

  - `executor`: The Ginkgo backend executor: `:omp` (OpenMP CPU), `:cuda` (NVIDIA GPU)
    or `:reference` (single-threaded). Defaults to `:omp`.
  - `kwargs...`: stored on the algorithm; not currently forwarded to Ginkgo. Likewise
    `args...` are stored and unused.

The solve uses `maxiters` and `reltol` from the cache; `abstol` and preconditioners are
not used. Ginkgo.jl currently only supports `Float32` with `Int32` indices, so the
matrix and vectors are converted to `Float32` on every solve.

!!! note

    Using this solver requires adding the package Ginkgo.jl, i.e. `using Ginkgo`
"""
function GinkgoJL_CG end

"""
```julia
GinkgoJL_GMRES(args...; executor = :omp, kwargs...)
```

A GMRES solver via Ginkgo.jl for general non-symmetric systems. It is
`GinkgoJL(args...; KrylovAlg = :gmres, executor, kwargs...)`; see `GinkgoJL` for the
full description.

## Keyword Arguments

  - `executor`: The Ginkgo backend executor: `:omp` (OpenMP CPU), `:cuda` (NVIDIA GPU)
    or `:reference` (single-threaded). Defaults to `:omp`.
  - `kwargs...`: stored on the algorithm; not currently forwarded to Ginkgo. Likewise
    `args...` are stored and unused.

Once available, the solve will use `maxiters` and `reltol` from the cache like
`GinkgoJL_CG`, with the same `Float32`/`Int32` conversion of the inputs.

!!! note

    Using this solver requires adding the package Ginkgo.jl, i.e. `using Ginkgo`.
    GMRES is not yet exposed by Ginkgo.jl v1. This stub is provided for forward compatibility;
    an error will be raised at solve time until Ginkgo.jl adds GMRES support.
"""
function GinkgoJL_GMRES end

"""
    MetalLUFactorization(; throwerror = true, residualsafety = false)

A wrapper over Apple's Metal GPU library for LU factorization. On each fresh
factorization the dense CPU matrix is copied to an `MtlArray` and factored on the GPU
with Metal's `lu`; the resulting factors and pivots are copied back to the host, and
the back-solve for each right-hand side then runs on the CPU with `ldiv!`. Only the
factorization is offloaded, and the GPU array is allocated per factorization. This
solver targets Metal-capable Apple Silicon Macs.

## Keyword Arguments

  - `throwerror`: whether to throw an error at construction when not on an Apple
    platform or when the Metal extension is not loaded. Defaults to `true`.
  - `residualsafety`: intended to enable the post-solve residual check described for
    `LUFactorization`. Defaults to `false`. That check lives in the generic
    factorization `solve!`, which the Metal extension replaces with its own `solve!` for
    this algorithm, so the flag currently has no effect here.

## Requirements
Using this solver requires that Metal.jl is loaded: `using Metal`. The constructor also
errors on any non-Apple platform, even with Metal.jl loaded, unless
`throwerror = false`.

## Performance Notes
- Most efficient for large dense matrices where GPU acceleration of the factorization outweighs transfer costs
- Particularly effective on Apple Silicon Macs with unified memory

## Example
```julia
using Metal
alg = MetalLUFactorization()
sol = solve(prob, alg)
```
"""
struct MetalLUFactorization <: AbstractFactorization
    residualsafety::Bool
    function MetalLUFactorization(; throwerror = true, residualsafety::Bool = false)
        return @static if !Sys.isapple()
            if throwerror
                error("MetalLUFactorization is only available on Apple platforms")
            else
                return new(residualsafety)
            end
        else
            ext = Base.get_extension(@__MODULE__, :LinearSolveMetalExt)
            if ext === nothing && throwerror
                error("MetalLUFactorization requires that Metal.jl is loaded, i.e. `using Metal`")
            else
                return new(residualsafety)
            end
        end
    end
end

"""
    MetalOffload32MixedLUFactorization(; throwerror = true)

A mixed precision Metal GPU-accelerated LU factorization that converts matrices to Float32
(ComplexF32 for complex inputs) before offloading to Metal GPU for factorization, then
converts back for the solve. This can provide speedups on Apple Silicon when reduced
precision is acceptable.

## Keyword Arguments

  - `throwerror`: whether to throw an error at construction when not on an Apple
    platform or when the Metal extension is not loaded. Defaults to `true`.

## Performance Notes
- Converts Float64 matrices to Float32 for GPU factorization
- Can be significantly faster for large matrices where memory bandwidth is limiting
- Particularly effective on Apple Silicon Macs with unified memory architecture
- May have reduced accuracy compared to full precision methods

## Requirements
Using this solver requires that Metal.jl is loaded: `using Metal`, and an Apple
platform.

## Example
```julia
using Metal
alg = MetalOffload32MixedLUFactorization()
sol = solve(prob, alg)
```
"""
struct MetalOffload32MixedLUFactorization <: AbstractFactorization
    function MetalOffload32MixedLUFactorization(; throwerror = true)
        return @static if !Sys.isapple()
            if throwerror
                error("MetalOffload32MixedLUFactorization is only available on Apple platforms")
            else
                return new()
            end
        else
            ext = Base.get_extension(@__MODULE__, :LinearSolveMetalExt)
            if ext === nothing && throwerror
                error("MetalOffload32MixedLUFactorization requires that Metal.jl is loaded, i.e. `using Metal`")
            else
                return new()
            end
        end
    end
end

"""
    BLISLUFactorization()

An LU factorization implementation using the BLIS (BLAS-like Library Instantiation Software)
framework. BLIS provides high-performance dense linear algebra kernels optimized for various
CPU architectures.

## Requirements
Using this solver requires that blis_jll is available and the BLIS extension is loaded.
The solver will be automatically available when conditions are met.

## Performance Notes
- Optimized for modern CPU architectures with BLIS-specific optimizations
- May provide better performance than standard BLAS on certain processors
- Best suited for dense matrices with Float32, Float64, ComplexF32, or ComplexF64 elements

## Example
```julia
alg = BLISLUFactorization()
sol = solve(prob, alg)
```
"""
struct BLISLUFactorization <: AbstractFactorization
    residualsafety::Bool
    function BLISLUFactorization(; throwerror = true, residualsafety::Bool = false)
        ext = Base.get_extension(@__MODULE__, :LinearSolveBLISExt)
        if ext === nothing && throwerror
            error("BLISLUFactorization requires that the BLIS extension is loaded and blis_jll is available")
        else
            return new(residualsafety)
        end
    end
end

"""
`CUSOLVERRFFactorization(; symbolic = :RF, reuse_symbolic = true)`

A GPU-accelerated sparse LU factorization using NVIDIA's cusolverRF library.
This solver is specifically designed for sparse matrices on CUDA GPUs and
provides high-performance factorization and solve capabilities.

## Keyword Arguments

  - `symbolic`: The symbolic factorization method to use. Options are:
    - `:RF` (default): Use cusolverRF's built-in symbolic analysis
    - `:KLU`: Use KLU for symbolic analysis
  - `reuse_symbolic`: Whether to reuse the symbolic factorization when the
    sparsity pattern doesn't change (default: `true`)

!!! note
    This solver requires CUSOLVERRF.jl to be loaded and only supports
    `Float64` element types with `Int32` indices.
"""
struct CUSOLVERRFFactorization <: AbstractSparseFactorization
    symbolic::Symbol
    reuse_symbolic::Bool

    function CUSOLVERRFFactorization(; symbolic::Symbol = :RF, reuse_symbolic::Bool = true)
        ext = Base.get_extension(@__MODULE__, :LinearSolveCUSOLVERRFExt)
        if ext === nothing
            error("CUSOLVERRFFactorization requires that CUSOLVERRF.jl is loaded, i.e. `using CUSOLVERRF`")
        else
            return new{}(symbolic, reuse_symbolic)
        end
    end
end

"""
    MKL32MixedLUFactorization()

A mixed precision LU factorization using Intel MKL that performs factorization in Float32
precision while maintaining Float64 interface. This can provide significant speedups
for large matrices when reduced precision is acceptable.

## Performance Notes
- Converts Float64 matrices to Float32 for factorization
- Uses optimized MKL routines for the factorization
- Can be 2x faster than full precision for memory-bandwidth limited problems
- May have reduced accuracy compared to full Float64 precision

## Requirements
This solver requires MKL to be available through MKL_jll.

## Example
```julia
alg = MKL32MixedLUFactorization()
sol = solve(prob, alg)
```
"""
struct MKL32MixedLUFactorization <: AbstractDenseFactorization end

"""
    AppleAccelerate32MixedLUFactorization()

A mixed precision LU factorization using Apple's Accelerate framework that performs
factorization in Float32 precision while maintaining Float64 interface. This can
provide significant speedups on Apple hardware when reduced precision is acceptable.

## Performance Notes
- Converts Float64 matrices to Float32 for factorization
- Uses optimized Accelerate routines for the factorization
- Particularly effective on Apple Silicon with unified memory
- May have reduced accuracy compared to full Float64 precision

## Requirements
This solver is only available on Apple platforms and requires the Accelerate framework.

## Example
```julia
alg = AppleAccelerate32MixedLUFactorization()
sol = solve(prob, alg)
```
"""
struct AppleAccelerate32MixedLUFactorization <: AbstractDenseFactorization end

"""
    OpenBLAS32MixedLUFactorization()

A mixed precision LU factorization using OpenBLAS that performs factorization in Float32
precision while maintaining Float64 interface. This can provide significant speedups
for large matrices when reduced precision is acceptable.

## Performance Notes
- Converts Float64 matrices to Float32 for factorization
- Uses optimized OpenBLAS routines for the factorization
- Can be 2x faster than full precision for memory-bandwidth limited problems
- May have reduced accuracy compared to full Float64 precision

## Requirements
This solver requires OpenBLAS to be available through OpenBLAS_jll.

## Example
```julia
alg = OpenBLAS32MixedLUFactorization()
sol = solve(prob, alg)
```
"""
struct OpenBLAS32MixedLUFactorization <: AbstractDenseFactorization end

"""
    RF32MixedLUFactorization(; pivot = Val(true), thread = Val(true), throwerror = true)
    RF32MixedLUFactorization(pivot::Val, thread::Val; throwerror = true)

A mixed precision LU factorization using RecursiveFactorization.jl that performs
factorization in Float32 precision while maintaining Float64 interface. This combines
the speed benefits of RecursiveFactorization.jl with reduced precision computation
for additional performance gains.

## Type Parameters
- `P`: Pivoting strategy as `Val{Bool}`. `Val{true}` enables partial pivoting for stability.
- `T`: Threading strategy as `Val{Bool}`. `Val{true}` enables multi-threading for performance.

## Constructor Arguments
- `pivot = Val(true)`: Enable partial pivoting. Set to `Val{false}` to disable for speed
  at the cost of numerical stability.
- `thread = Val(true)`: Enable multi-threading. Set to `Val{false}` for single-threaded
  execution.
- `throwerror = true`: Whether to throw an error if RecursiveFactorization.jl is not loaded.

## Performance Notes
- Converts Float64 matrices to Float32 for factorization
- Leverages RecursiveFactorization.jl's optimized blocking strategies
- Can provide significant speedups for small to medium matrices (< 500×500)
- May have reduced accuracy compared to full Float64 precision

## Requirements
Using this solver requires that RecursiveFactorization.jl is loaded: `using RecursiveFactorization`

## Example
```julia
using RecursiveFactorization
# Fast mixed precision with pivoting
alg1 = RF32MixedLUFactorization()
# Fastest mixed precision (no pivoting), less stable
alg2 = RF32MixedLUFactorization(pivot=Val(false))
```
"""
struct RF32MixedLUFactorization{P, T} <: AbstractDenseFactorization
    function RF32MixedLUFactorization(::Val{P}, ::Val{T}; throwerror = true) where {P, T}
        if !userecursivefactorization(nothing)
            throwerror &&
                error("RF32MixedLUFactorization requires that RecursiveFactorization.jl is loaded, i.e. `using RecursiveFactorization`")
        end
        return new{P, T}()
    end
end

function RF32MixedLUFactorization(; pivot = Val(true), thread = Val(true), throwerror = true)
    return RF32MixedLUFactorization(pivot, thread; throwerror)
end

"""
    AlgebraicMultigridJL(args...; kwargs...)

A wrapper for [AlgebraicMultigrid.jl](https://github.com/JuliaLinearAlgebra/AlgebraicMultigrid.jl)
solvers. The AMG hierarchy is built with `SciMLBase.init(amg_alg, A, b; kwargs...)` on
each fresh `A` and then used as a standalone iterative solver via `solve!`. It is meant
for square sparse systems of the kind algebraic multigrid handles well, typically
symmetric positive definite or M-matrix-like discretizations of elliptic PDEs; a
non-square `A` fails an assertion at `init`.

## Positional Arguments

The first positional argument (if given) is the AMG algorithm type. If omitted,
defaults to `AlgebraicMultigrid.RugeStubenAMG()`.

## Keyword Arguments

All keyword arguments are forwarded to the AMG hierarchy constructor.

Only `reltol` and `maxiters` from the cache are used (as `reltol` and `maxiter` of the
AMG `solve!`); `abstol` and any `Pl`/`Pr` preconditioners are not used. The returned
solution always reports `ReturnCode.Success`, so check the residual yourself if
convergence matters.

## Example

```julia
using LinearSolve, AlgebraicMultigrid
# Default (Ruge-Stuben)
alg = AlgebraicMultigridJL()
# Smoothed Aggregation
alg = AlgebraicMultigridJL(AlgebraicMultigrid.SmoothedAggregationAMG())
```

!!! note

    Using this solver requires adding the package AlgebraicMultigrid.jl,
    i.e. `using AlgebraicMultigrid`
"""
struct AlgebraicMultigridJL{A, K} <: SciMLLinearSolveAlgorithm
    args::A
    kwargs::K
end

function AlgebraicMultigridJL(args...; kwargs...)
    return AlgebraicMultigridJL(args, kwargs)
end

needs_concrete_A(::AlgebraicMultigridJL) = true

# The AMG solve reads `cache.reltol` on every `solve!`.
update_tolerances_internal!(cache, ::AlgebraicMultigridJL, abstol, reltol) = nothing

"""
`ParUFactorization(;reuse_symbolic=true)`

A parallel sparse LU factorization from SuiteSparse's
[ParU](https://github.com/DrTimothyAldenDavis/SuiteSparse) library.
ParU is a multithreaded direct solver for sparse systems of linear equations
using OpenMP task parallelism for the numeric factorization phase.

ParU calls UMFPACK for its symbolic analysis phase (computing fill-reducing
column ordering and symbolic factorization), then performs a parallel numeric
factorization exploiting dense frontal matrices. It can outperform UMFPACK on
larger systems where the parallelism can be exploited.

Only supports `Float64` element type.

## Keyword Arguments

  - `reuse_symbolic`: Cache and reuse the symbolic factorization across solves
    when the sparsity pattern of `A` does not change. Defaults to `true`.

!!! note

    Using this solver requires loading the package `ParU_jll`, i.e.:
    ```julia
    import ParU_jll
    using LinearSolve, SparseArrays
    ```

## Example

```julia
import ParU_jll
using LinearSolve, SparseArrays

A = sprand(100, 100, 0.1) + 10I
b = rand(100)
prob = LinearProblem(A, b)
sol = solve(prob, ParUFactorization())
```
"""
struct ParUFactorization <: AbstractSparseFactorization
    reuse_symbolic::Bool
    function ParUFactorization(; reuse_symbolic::Bool = true)
        ext = Base.get_extension(@__MODULE__, :LinearSolveParUExt)
        if ext === nothing
            error("ParUFactorization requires that ParU_jll and SparseArrays are loaded, i.e. `import ParU_jll; using SparseArrays`")
        end
        return new(reuse_symbolic)
    end
end

"""
`MUMPSFactorization(; sym = :unsymmetric, transposed = false, verbose = false, ooc = false, itref = 0, user_perm = false, icntl = nothing, cntl = nothing, par = 1)`

A sparse direct solver wrapper around [MUMPS.jl](https://github.com/JuliaSmoothOptimizers/MUMPS.jl),
backed by the `MUMPS_jll` artifact.

This wrapper is intended for repeated solves with the same sparse matrix, using a cached
MUMPS factorization inside the `LinearSolve` cache. When `cache.A` is replaced, the old
MUMPS object is finalized and a new factorization is built.

## Keyword Arguments

  - `sym`: Matrix structure passed to MUMPS. Choices are `:unsymmetric` (default),
    `:definite`, and `:symmetric`.
  - `transposed`: Solve `A' * x = b` instead of `A * x = b`.
  - `verbose`: Enable MUMPS output.
  - `ooc`: Enable out-of-core factorization.
  - `itref`: Maximum number of iterative refinement steps.
  - `user_perm`: Tell MUMPS to use a user-supplied permutation.
  - `icntl`: Optional custom MUMPS `ICNTL` vector. When provided, it overrides the
    wrapper-generated control vector.
  - `cntl`: Optional custom MUMPS `CNTL` vector.
  - `par`: MUMPS host participation flag. Defaults to `1`.

!!! note

    Using this solver requires loading `MUMPS.jl` and `SparseArrays`, and initializing MPI:
    ```julia
    using MPI, MUMPS, SparseArrays
    MPI.Init()
    ```

## Supported Element Types

`Float32`, `Float64`, `ComplexF32`, and `ComplexF64`.

Inputs with other element types are rejected with an error instead of being
silently converted to a lower-precision type.

## Memory Management

MUMPS holds MPI-backed resources outside Julia's GC. Call
`cleanup_mumps_cache!` explicitly when you are finished with a solve and before
`MPI.Finalize()`:

```julia
using LinearSolve, SparseArrays, MPI, MUMPS

MPI.Init()
MUMPSExt = Base.get_extension(LinearSolve, :LinearSolveMUMPSExt)

A = sparse([4.0 1.0; 2.0 3.0])
b = [1.0, 2.0]
cache = SciMLBase.init(LinearProblem(A, b), MUMPSFactorization())
sol = solve!(cache)

MUMPSExt.cleanup_mumps_cache!(cache)
MPI.Finalize()
```

The extension also registers a GC finalizer as a safety net, but explicit
cleanup is strongly preferred for deterministic teardown.

## Example

```julia
using LinearSolve, SparseArrays, MPI, MUMPS

MPI.Init()
MUMPSExt = Base.get_extension(LinearSolve, :LinearSolveMUMPSExt)
A = sparse([4.0 1.0; 2.0 3.0])
b = [1.0, 2.0]
cache = SciMLBase.init(LinearProblem(A, b), MUMPSFactorization())
sol = solve!(cache)
MUMPSExt.cleanup_mumps_cache!(cache)
```
"""
struct MUMPSFactorization <: AbstractSparseFactorization
    sym::Int
    transposed::Bool
    verbose::Bool
    ooc::Bool
    itref::Int
    user_perm::Bool
    icntl::Any
    cntl::Any
    par::Int

    function MUMPSFactorization(;
            sym::Union{Symbol, Integer} = :unsymmetric,
            transposed::Bool = false,
            verbose::Bool = false,
            ooc::Bool = false,
            itref::Int = 0,
            user_perm::Bool = false,
            icntl = nothing,
            cntl = nothing,
            par::Int = 1,
        )
        ext = Base.get_extension(@__MODULE__, :LinearSolveMUMPSExt)
        if ext === nothing
            error("MUMPSFactorization requires that MUMPS and SparseArrays are loaded, i.e. `using MPI, MUMPS, SparseArrays`")
        end
        sym_val = if sym isa Symbol
            if sym === :unsymmetric
                0
            elseif sym === :definite
                1
            elseif sym === :symmetric
                2
            else
                error("Unknown MUMPS symmetry flag: $sym")
            end
        else
            Int(sym)
        end
        return new(sym_val, transposed, verbose, ooc, itref, user_perm, icntl, cntl, par)
    end
end

"""
`SuperLUDISTFactorization(; comm = nothing, nprow = 0, npcol = 0, options = nothing, threads = nothing)`

A sparse direct solver wrapper around
[SuperLUDIST.jl](https://github.com/JuliaSparse/SuperLUDIST.jl), backed by the
`SuperLU_DIST_jll` artifact through that package.

This wrapper targets ordinary replicated Julia sparse matrices
(`SparseMatrixCSC`). Each participating rank builds the same Julia matrix and
right-hand side locally, and `SuperLUDIST` performs the distributed solve on
the requested communicator. The resulting factorization is cached inside the
`LinearSolve` cache and reused across repeated solves with new right-hand
sides.

## Keyword Arguments

  - `comm`: optional MPI communicator. `nothing` maps to `MPI.COMM_SELF`.
  - `nprow`, `npcol`: process-grid dimensions for SuperLU_DIST. If both are
    left as `0`, a near-square grid is chosen automatically from the
    communicator size.
  - `options`: either `nothing`, a `NamedTuple` of `SuperLUDIST.Options` field
    updates, or a prebuilt `SuperLUDIST.Options` object.
  - `threads`: optional OpenMP thread count passed through
    `SuperLUDIST.superlu_set_num_threads`.

!!! note

    Using this solver requires loading `SuperLUDIST.jl` and `SparseArrays`, and
    initializing MPI for multi-rank usage:
    ```julia
    using MPI, SuperLUDIST, SparseArrays
    MPI.Init()
    ```

## Supported Element Types

`Float32` and `Float64`.

`ComplexF64` is intentionally rejected for now because the current upstream
`SuperLUDIST.jl` replicated complex solve path crashes during finalization.

## Example

```julia
using LinearSolve, SparseArrays, MPI, SuperLUDIST

MPI.Init()
A = sparse([4.0 1.0; 2.0 3.0])
b = [1.0, 2.0]

sol = solve(
    LinearProblem(A, b),
    SuperLUDISTFactorization(; comm = MPI.COMM_SELF)
)
```
"""
struct SuperLUDISTFactorization <: AbstractSparseFactorization
    comm::Any
    nprow::Int
    npcol::Int
    options::Any
    threads::Union{Nothing, Int}

    function SuperLUDISTFactorization(;
            comm = nothing,
            nprow::Integer = 0,
            npcol::Integer = 0,
            options = nothing,
            threads::Union{Nothing, Integer} = nothing,
        )
        ext = Base.get_extension(@__MODULE__, :LinearSolveSuperLUDISTExt)
        if ext === nothing
            error("SuperLUDISTFactorization requires that SuperLUDIST and SparseArrays are loaded, i.e. `using MPI, SuperLUDIST, SparseArrays`")
        end
        nprow >= 0 || error("nprow must be nonnegative")
        npcol >= 0 || error("npcol must be nonnegative")
        threads === nothing || threads > 0 || error("threads must be positive")
        return new(comm, Int(nprow), Int(npcol), options, threads === nothing ? nothing : Int(threads))
    end
end

"""
`HSLMA57Factorization(; kwargs...)`

A sparse symmetric direct solver powered by
[HSL.jl](https://github.com/JuliaSmoothOptimizers/HSL.jl)'s MA57 backend.

Keyword arguments are forwarded to `HSL.Ma57(...)`.

!!! note

    Using this solver requires loading `HSL.jl` and `SparseArrays`:
    ```julia
    using HSL, SparseArrays
    ```

    `HSL.jl` requires a manual installation of `HSL_jll.jl` due proprietary
    licensing. See `HSL.jl` installation instructions:
    https://github.com/JuliaSmoothOptimizers/HSL.jl
"""
struct HSLMA57Factorization{K} <: AbstractSparseFactorization
    kwargs::K

    function HSLMA57Factorization(; kwargs...)
        ext = Base.get_extension(@__MODULE__, :LinearSolveHSLExt)
        if ext === nothing
            error("HSLMA57Factorization requires `using HSL, SparseArrays`. Note: `HSL.jl` requires manual installation of `HSL_jll.jl`.")
        end
        return new{typeof(kwargs)}(kwargs)
    end
end

"""
`HSLMA97Factorization(; matrix_type = :real_indef, kwargs...)`

A sparse symmetric/Hermitian direct solver powered by
[HSL.jl](https://github.com/JuliaSmoothOptimizers/HSL.jl)'s MA97 backend.

## Keyword Arguments

  - `matrix_type`: Passed to `HSL.ma97_factorize!`. Supported values include
    `:real_spd`, `:real_indef`, `:herm_pd`, `:herm_indef`, `:cmpl_indef`.
  - `kwargs...`: Forwarded to `HSL.Ma97(...)`.

!!! note

    Using this solver requires loading `HSL.jl` and `SparseArrays`:
    ```julia
    using HSL, SparseArrays
    ```

    `HSL.jl` requires a manual installation of `HSL_jll.jl` due proprietary
    licensing. See `HSL.jl` installation instructions:
    https://github.com/JuliaSmoothOptimizers/HSL.jl
"""
struct HSLMA97Factorization{K} <: AbstractSparseFactorization
    matrix_type::Symbol
    kwargs::K

    function HSLMA97Factorization(; matrix_type::Symbol = :real_indef, kwargs...)
        ext = Base.get_extension(@__MODULE__, :LinearSolveHSLExt)
        if ext === nothing
            error("HSLMA97Factorization requires `using HSL, SparseArrays`. Note: `HSL.jl` requires manual installation of `HSL_jll.jl`.")
        end

        matrix_type ∈ (:real_spd, :real_indef, :herm_pd, :herm_indef, :cmpl_indef) || error(
            "Unsupported matrix_type: $(matrix_type). Expected one of :real_spd, :real_indef, :herm_pd, :herm_indef, :cmpl_indef."
        )

        return new{typeof(kwargs)}(matrix_type, kwargs)
    end
end

"""
    ElementalJL(; method = :LU)

A wrapper for [Elemental.jl](https://github.com/JuliaParallel/Elemental.jl),
providing distributed-memory dense linear algebra solvers built on the
[Elemental](https://github.com/elemental/Elemental) C++ library by Jack Poulson.
(LLNL maintains an active GPU-focused fork of Elemental called
[Hydrogen](https://github.com/LLNL/Elemental).)

## Keyword Arguments

  - `method`: The factorization method to use. Options:
    - `:LU` (default) — LU factorization with partial pivoting. Suitable for
      general square systems.
    - `:QR` — QR factorization. Suitable for square or overdetermined systems.
    - `:LQ` — LQ factorization. Suitable for square or underdetermined systems.
    - `:Cholesky` — Cholesky factorization. Requires the matrix to be Hermitian
      positive definite.

## Supported Element Types

`Float32`, `Float64`, `ComplexF32`, `ComplexF64`. Matrices with other element
types are promoted to `Float64` (real) or `ComplexF64` (complex) before being
passed to Elemental.

## Notes

  - Serial `Elemental.Matrix` values are accepted directly as the problem
    matrix `A`; they are copied before factorization so the original is
    never mutated.
  - When `A` is a standard Julia `AbstractMatrix`, it is copied into an
    `Elemental.Matrix` for the factorization.
  - The factorization is cached across repeated solves with the same matrix
    (i.e. when `isfresh = false`).

!!! note

    Using this solver requires adding the package Elemental.jl:
    ```julia
    using Elemental
    ```
    Elemental.jl automatically initialises MPI when loaded; no explicit
    `MPI.Init()` call is needed for serial usage.

## Example

```julia
using LinearSolve, Elemental

A = rand(100, 100); A = A + A' + 100I  # well-conditioned
b = rand(100)
prob = LinearProblem(A, b)

# LU (default)
sol = solve(prob, ElementalJL())
# LQ
sol = solve(prob, ElementalJL(method = :LQ))
# Cholesky (symmetric positive definite)
sol = solve(prob, ElementalJL(method = :Cholesky))
```
"""
struct ElementalJL <: AbstractDenseFactorization
    method::Symbol
end

function ElementalJL(; method::Symbol = :LU)
    return ElementalJL(method)
end

"""
    SpecializedLUFactorization()

A type-stable, structure-detecting dense LU-style solver from
SpecializingFactorizations.jl. It cheaply scans the (dense) matrix `A` to detect
whether it actually has special structure (diagonal, bidiagonal, tridiagonal,
banded, triangular, symmetric positive definite, symmetric/Hermitian indefinite)
and dispatches to the matching specialized factorization instead of always using
a general `O(n^3)` LU. Detection is tracked by a runtime enum stored in a single
concrete workspace type, so the whole detect -> factor -> solve pipeline is
type-stable and allocation-free on the warm path.

This is for **square** systems only; use [`SpecializedQRFactorization`](@ref) for
rectangular / rank-deficient least-squares problems.

!!! note

    Using this solver requires that SpecializingFactorizations.jl is loaded:
    `using SpecializingFactorizations`.

## Example

```julia
using SpecializingFactorizations
A = Matrix(Tridiagonal(rand(99), rand(100) .+ 4, rand(99)))
b = rand(100)
prob = LinearProblem(A, b)
sol = solve(prob, SpecializedLUFactorization())
```
"""
struct SpecializedLUFactorization <: AbstractDenseFactorization end

"""
    SpecializedQRFactorization()

A type-stable, rank-revealing dense QR least-squares solver from
SpecializingFactorizations.jl. It uses a column-pivoted, rank-revealing QR
(LAPACK `geqp3`) to reveal the numerical rank of a possibly **rectangular** or
**rank-deficient** matrix and returns the least-squares solution for any shape,
**including singular and rank-deficient** matrices, without ever throwing. For a
rank-deficient system it returns the minimum-norm least-squares solution
(matching `pinv(A) * b`). For square `BlasFloat` inputs it reuses the same
structure-detection scan as [`SpecializedLUFactorization`](@ref) to take cheaper
structured paths when doing so provably reproduces the dense rank-revealing
result.

!!! note

    Using this solver requires that SpecializingFactorizations.jl is loaded:
    `using SpecializingFactorizations`.

## Example

```julia
using SpecializingFactorizations
A = randn(100, 40)  # overdetermined least-squares
b = randn(100)
prob = LinearProblem(A, b)
sol = solve(prob, SpecializedQRFactorization())
```
"""
struct SpecializedQRFactorization <: AbstractDenseFactorization end
