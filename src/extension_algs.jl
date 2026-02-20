# This file only include the algorithm struct to be exported by LinearSolve.jl. The main
# functionality is implemented as package extensions
"""
    PETScAlgorithm(solver_type = :gmres; kwargs...)

A `LinearSolve.jl` algorithm that wraps the
[PETSc.jl](https://github.com/JuliaParallel/PETSc.jl) interface to PETSc’s
KSP (Krylov Subspace) linear solvers.

This algorithm enables the use of PETSc’s scalable iterative solvers and
preconditioners from within the SciML linear solve ecosystem, supporting both
serial and MPI-parallel workflows while preserving the `LinearSolve.solve`
API.

`PETScAlgorithm` constructs and manages PETSc matrices, vectors, KSP solvers,
and optional null-space objects behind the scenes. It can be used as a drop-in
replacement for other `SciMLLinearSolveAlgorithm`s when access to PETSc
features such as algebraic multigrid, domain decomposition, or advanced solver
monitoring is required.

!!! compat
    Requires Julia ≥ 1.10 with `PETSc.jl` and `MPI.jl` loaded:
    using PETSc, MPI

---

## Capabilities

* Access to PETSc Krylov solvers (GMRES, CG, BiCGStab, Richardson, etc.)
* Integration with PETSc preconditioners (ILU, LU, GAMG, Hypre, …)
* Serial and MPI-distributed solves using the same API
* Support for null-space specification (e.g. pressure Poisson problems)
* Optional separate preconditioner matrix
* Forwarding of arbitrary PETSc Options Database flags
* Solve of transposed systems via PETSc
* Reuse of initial guesses across nonlinear or time-stepping workflows

---

## Memory Management

PETSc objects live in C-side memory that Julia’s garbage collector does not
track.

Always call `cleanup_petsc_cache!` after finishing with a solution, cache, or
algorithm instance:

    sol = solve(prob, PETScAlgorithm(:gmres))
    cleanup_petsc_cache!(sol)

A GC finalizer is registered as a safety net, but deterministic cleanup is
strongly recommended.

Access the cleanup helper via the extension module:

```julia
PETScExt = Base.get_extension(LinearSolve, :LinearSolvePETScExt)
PETScExt.cleanup_petsc_cache!(sol)
````

!!! warning "MPI cleanup"
    When using multiple MPI ranks, cleanup is collective — all ranks must
    call it together.

---

## Positional Arguments

- `solver_type::Symbol`: PETSc KSP solver type. Common values include
  `:gmres` (default), `:cg`, `:bicg`, `:bcgs`, `:preonly`, and `:richardson`.

---

## Keyword Arguments

- `pc_type::Symbol`
  PETSc preconditioner type. Examples: `:jacobi`, `:ilu`, `:lu`, `:gamg`,
  `:hypre`, `:none`.

- `comm`
  MPI communicator.
  Defaults to `nothing`, which maps to `MPI.COMM_SELF` at solve time (serial).
  Use `MPI.COMM_WORLD` for distributed solves.

  Each rank must own the full Julia matrix structure, but inserts only its
  locally owned rows into PETSc.

- `nullspace::Symbol`
  Null-space handling strategy:
  `:none`, `:constant`, or `:custom`.

- `nullspace_vecs`
  Vector of orthonormal null-space basis vectors used when
  `nullspace = :custom`.

- `prec_matrix`
  Optional matrix used only for constructing the preconditioner, allowing
  different operator and preconditioning discretizations.

- `initial_guess_nonzero::Bool`
  If `true`, PETSc uses the existing solution vector as the initial Krylov
  guess instead of zero.

- `transposed::Bool`
  Solve the transposed system `Aᵀx = b` using PETSc’s transpose solve.

- `ksp_options::NamedTuple`
  Additional PETSc Options Database flags forwarded automatically.
  Keys are converted to PETSc CLI-style flags.

---

## Common `ksp_options`

| Option | Description |
| :--- | :--- |
| `ksp_monitor = ""` | Print residual norm each iteration |
| `ksp_view = ""` | Print solver configuration summary |
| `ksp_rtol = 1e-12` | Relative convergence tolerance |
| `pc_factor_levels = 2` | ILU fill levels |
| `log_view = ""` | PETSc performance logging |

---

## Examples

```julia
using PETSc, MPI, SparseArrays, LinearSolve, LinearAlgebra

MPI.Init()

n = 100
A = sprand(n, n, 0.1); A = A + A' + 20I
b = rand(n)

prob = LinearProblem(A, b)

PETScExt = Base.get_extension(LinearSolve, :LinearSolvePETScExt)

# Serial solve
sol = solve(prob, PETScAlgorithm(:gmres; pc_type = :ilu,
                                 ksp_options = (ksp_monitor="",)))
PETScExt.cleanup_petsc_cache!(sol)

# MPI-parallel solve
sol = solve(prob, PETScAlgorithm(:gmres; pc_type = :gamg,
                                 comm = MPI.COMM_WORLD))
PETScExt.cleanup_petsc_cache!(sol)

```
"""
struct PETScAlgorithm <: SciMLLinearSolveAlgorithm
    solver_type           :: Symbol
    pc_type               :: Symbol
    comm                  :: Any        # MPI.Comm — stored as Any to avoid MPI dep in LinearSolve
    nullspace             :: Symbol     # :none | :constant | :custom
    nullspace_vecs        :: Any        # nothing | Vector{Vector{T}}
    prec_matrix           :: Any        # nothing | AbstractMatrix
    initial_guess_nonzero :: Bool
    transposed            :: Bool
    ksp_options           :: NamedTuple

    function PETScAlgorithm(
            solver_type           :: Symbol     = :gmres;
            pc_type               :: Symbol     = :none,
            comm                                = nothing,   # nothing → MPI.COMM_SELF at solve time
            nullspace             :: Symbol     = :none,
            nullspace_vecs                      = nothing,
            prec_matrix                         = nothing,
            initial_guess_nonzero :: Bool       = false,
            transposed            :: Bool       = false,
            ksp_options           :: NamedTuple = NamedTuple(),
        )
        ext = Base.get_extension(@__MODULE__, :LinearSolvePETScExt)
        if ext === nothing
            error("PETScAlgorithm requires that PETSc and MPI are loaded, \
                   i.e. `using PETSc, MPI`")
        end
        if nullspace ∉ (:none, :constant, :custom)
            error("nullspace must be :none, :constant, or :custom (got :$nullspace)")
        end
        if nullspace == :custom && nullspace_vecs === nothing
            error("nullspace = :custom requires nullspace_vecs to be provided")
        end
        return new(
            solver_type, pc_type,
            comm,
            nullspace, nullspace_vecs,
            prec_matrix,
            initial_guess_nonzero,
            transposed,
            ksp_options,
        )
    end
end

"""
`HYPREAlgorithm(solver; Pl = nothing)`

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

  - `Pl`: A choice of left preconditioner.

## Example

For example, to use `HYPRE.PCG` as the solver, with `HYPRE.BoomerAMG` as the preconditioner,
the algorithm should be defined as follows:

```julia
A, b = setup_system(...)
prob = LinearProblem(A, b)
alg = HYPREAlgorithm(HYPRE.PCG)
prec = HYPRE.BoomerAMG
sol = solve(prob, alg; Pl = prec)
```
"""
struct HYPREAlgorithm <: SciMLLinearSolveAlgorithm
    solver::Any
    function HYPREAlgorithm(solver)
        ext = Base.get_extension(@__MODULE__, :LinearSolveHYPREExt)
        if ext === nothing
            error("HYPREAlgorithm requires that HYPRE is loaded, i.e. `using HYPRE`")
        else
            return new{}(solver)
        end
    end
end

# Debug: About to define CudaOffloadLUFactorization
"""
`CudaOffloadLUFactorization()`

An offloading technique used to GPU-accelerate CPU-based computations using LU factorization.
Requires a sufficiently large `A` to overcome the data transfer costs.

!!! note

    Using this solver requires adding the package CUDA.jl, i.e. `using CUDA`
"""
struct CudaOffloadLUFactorization <: AbstractFactorization
    function CudaOffloadLUFactorization(; throwerror = true)
        ext = Base.get_extension(@__MODULE__, :LinearSolveCUDAExt)
        if ext === nothing && throwerror
            error("CudaOffloadLUFactorization requires that CUDA is loaded, i.e. `using CUDA`")
        else
            return new()
        end
    end
end

"""
`CUDAOffload32MixedLUFactorization()`

A mixed precision GPU-accelerated LU factorization that converts matrices to Float32
before offloading to CUDA GPU for factorization, then converts back for the solve.
This can provide speedups when the reduced precision is acceptable and memory
bandwidth is a bottleneck.

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
    RFLUFactorization{P, T}(; pivot = Val(true), thread = Val(true))

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
    function RFLUFactorization(::Val{P}, ::Val{T}; throwerror = true) where {P, T}
        if !userecursivefactorization(nothing)
            throwerror &&
                error("RFLUFactorization requires that RecursiveFactorization.jl is loaded, i.e. `using RecursiveFactorization`")
        end
        return new{P, T}()
    end
end

function RFLUFactorization(; pivot = Val(true), thread = Val(true), throwerror = true)
    return RFLUFactorization(pivot, thread; throwerror)
end

"""
`ButterflyFactorization()`

A fast pure Julia LU-factorization implementation
using RecursiveFactorization.jl. This method utilizes a butterfly
factorization approach rather than pivoting.
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
    FastQRFactorization{P}(; pivot = ColumnNorm(), blocksize = 36)

A high-performance QR factorization using the FastLapackInterface.jl package.
This provides an optimized interface to LAPACK QR routines with reduced overhead
compared to the standard LinearAlgebra LAPACK wrappers.

## Type Parameters
- `P`: The type of pivoting strategy used

## Fields
- `pivot::P`: Pivoting strategy (e.g., `ColumnNorm()` for column pivoting, `nothing` for no pivoting)
- `blocksize::Int`: Block size for the blocked QR algorithm (default: 36)

## Features
- Reduced function call overhead compared to standard LAPACK wrappers
- Supports various pivoting strategies for numerical stability
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
using FastLapackInterface
# QR with column pivoting
alg1 = FastQRFactorization()
# QR without pivoting for speed
alg2 = FastQRFactorization(pivot=nothing)
# Custom block size
alg3 = FastQRFactorization(blocksize=64)
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

A sparse factorization method using MKL Pardiso.

!!! note

    Using this solver requires adding the package Pardiso.jl, i.e. `using Pardiso`

## Keyword Arguments

Setting `cache_analysis = true` disables Pardiso's scaling and matching defaults
and caches the result of the initial analysis phase for all further computations
with this solver.

For the definition of the other keyword arguments, see the Pardiso.jl documentation.
All values default to `nothing` and the solver internally determines the values
given the input types, and these keyword arguments are only for overriding the
default handling process. This should not be required by most users.
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

A mixed factorization+iterative method using MKL Pardiso.

!!! note

    Using this solver requires adding the package Pardiso.jl, i.e. `using Pardiso`

## Keyword Arguments

Setting `cache_analysis = true` disables Pardiso's scaling and matching defaults
and caches the result of the initial analysis phase for all further computations
with this solver.

For the definition of the other keyword arguments, see the Pardiso.jl documentation.
All values default to `nothing` and the solver internally determines the values
given the input types, and these keyword arguments are only for overriding the
default handling process. This should not be required by most users.
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

A sparse factorization method using Panua Pardiso.

!!! note

    Using this solver requires adding the package Pardiso.jl, i.e. `using Pardiso`

## Keyword Arguments

Setting `cache_analysis = true` disables Pardiso's scaling and matching defaults
and caches the result of the initial analysis phase for all further computations
with this solver.

For the definition of the keyword arguments, see the Pardiso.jl documentation.
All values default to `nothing` and the solver internally determines the values
given the input types, and these keyword arguments are only for overriding the
default handling process. This should not be required by most users.
"""
PanuaPardisoFactorize(; kwargs...) = PardisoJL(;
    vendor = :Panua, solver_type = 0, kwargs...
)

"""
```julia
PanuaPardisoIterate(; nprocs::Union{Int, Nothing} = nothing,
    matrix_type = nothing,
    iparm::Union{Vector{Tuple{Int, Int}}, Nothing} = nothing,
    dparm::Union{Vector{Tuple{Int, Int}}, Nothing} = nothing)
```

A mixed factorization+iterative method using Panua Pardiso.

!!! note

    Using this solver requires adding the package Pardiso.jl, i.e. `using Pardiso`

## Keyword Arguments

For the definition of the keyword arguments, see the Pardiso.jl documentation.
All values default to `nothing` and the solver internally determines the values
given the input types, and these keyword arguments are only for overriding the
default handling process. This should not be required by most users.
"""
PanuaPardisoIterate(; kwargs...) = PardisoJL(; vendor = :Panua, solver_type = 1, kwargs...)

"""
```julia
PardisoJL(; nprocs::Union{Int, Nothing} = nothing,
    solver_type = nothing,
    matrix_type = nothing,
    iparm::Union{Vector{Tuple{Int, Int}}, Nothing} = nothing,
    dparm::Union{Vector{Tuple{Int, Int}}, Nothing} = nothing,
    vendor::Union{Symbol, Nothing} = nothing
)
```

A generic method using  Pardiso. Specifying `solver_type` is required.

!!! note

    Using this solver requires adding the package Pardiso.jl, i.e. `using Pardiso`

## Keyword Arguments

The `vendor` keyword allows to choose between Panua pardiso  (former pardiso-project.org; `vendor=:Panua`)
and  MKL Pardiso (`vendor=:MKL`). If `vendor==nothing`, Panua pardiso is preferred over MKL Pardiso.

For the definition of the other keyword arguments, see the Pardiso.jl documentation.
All values default to `nothing` and the solver internally determines the values
given the input types, and these keyword arguments are only for overriding the
default handling process. This should not be required by most users.
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
KrylovKitJL(args...; KrylovAlg = Krylov.gmres!, kwargs...)
```

A generic iterative solver implementation allowing the choice of KrylovKit.jl
solvers.

!!! note

    Using this solver requires adding the package KrylovKit.jl, i.e. `using KrylovKit`
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
KrylovKitJL_CG(args...; Pl = nothing, Pr = nothing, kwargs...)
```

A generic CG implementation for Hermitian and positive definite linear systems

!!! note

    Using this solver requires adding the package KrylovKit.jl, i.e. `using KrylovKit`
"""
function KrylovKitJL_CG end

"""
```julia
KrylovKitJL_GMRES(args...; Pl = nothing, Pr = nothing, gmres_restart = 0, kwargs...)
```

A generic GMRES implementation.

!!! note

    Using this solver requires adding the package KrylovKit.jl, i.e. `using KrylovKit`
"""
function KrylovKitJL_GMRES end

"""
```julia
IterativeSolversJL(args...;
    generate_iterator = IterativeSolvers.gmres_iterable!,
    Pl = nothing, Pr = nothing,
    gmres_restart = 0, kwargs...)
```

A generic wrapper over the IterativeSolvers.jl solvers.

!!! note

    Using this solver requires adding the package IterativeSolvers.jl, i.e. `using IterativeSolvers`
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
IterativeSolversJL_CG(args...; Pl = nothing, Pr = nothing, kwargs...)
```

A wrapper over the IterativeSolvers.jl CG.

!!! note

    Using this solver requires adding the package IterativeSolvers.jl, i.e. `using IterativeSolvers`
"""
function IterativeSolversJL_CG end

"""
```julia
IterativeSolversJL_GMRES(args...; Pl = nothing, Pr = nothing, gmres_restart = 0, kwargs...)
```

A wrapper over the IterativeSolvers.jl GMRES.

!!! note

    Using this solver requires adding the package IterativeSolvers.jl, i.e. `using IterativeSolvers`
"""
function IterativeSolversJL_GMRES end

"""
```julia
IterativeSolversJL_IDRS(args...; Pl = nothing, kwargs...)
```

A wrapper over the IterativeSolvers.jl IDR(S).

!!! note

    Using this solver requires adding the package IterativeSolvers.jl, i.e. `using IterativeSolvers`
"""
function IterativeSolversJL_IDRS end

"""
```julia
IterativeSolversJL_BICGSTAB(args...; Pl = nothing, Pr = nothing, kwargs...)
```

A wrapper over the IterativeSolvers.jl BICGSTAB.

!!! note

    Using this solver requires adding the package IterativeSolvers.jl, i.e. `using IterativeSolvers`
"""
function IterativeSolversJL_BICGSTAB end

"""
```julia
IterativeSolversJL_MINRES(args...; Pl = nothing, Pr = nothing, kwargs...)
```

A wrapper over the IterativeSolvers.jl MINRES.

!!! note

    Using this solver requires adding the package IterativeSolvers.jl, i.e. `using IterativeSolvers`
"""
function IterativeSolversJL_MINRES end

"""
    MetalLUFactorization()

A wrapper over Apple's Metal GPU library for LU factorization. Direct calls to Metal
in a way that pre-allocates workspace to avoid allocations and automatically offloads
to the GPU. This solver is optimized for Metal-capable Apple Silicon Macs.

## Requirements
Using this solver requires that Metal.jl is loaded: `using Metal`

## Performance Notes
- Most efficient for large dense matrices where GPU acceleration benefits outweigh transfer costs
- Automatically manages GPU memory and transfers
- Particularly effective on Apple Silicon Macs with unified memory

## Example
```julia
using Metal
alg = MetalLUFactorization()
sol = solve(prob, alg)
```
"""
struct MetalLUFactorization <: AbstractFactorization
    function MetalLUFactorization(; throwerror = true)
        return @static if !Sys.isapple()
            if throwerror
                error("MetalLUFactorization is only available on Apple platforms")
            else
                return new()
            end
        else
            ext = Base.get_extension(@__MODULE__, :LinearSolveMetalExt)
            if ext === nothing && throwerror
                error("MetalLUFactorization requires that Metal.jl is loaded, i.e. `using Metal`")
            else
                return new()
            end
        end
    end
end

"""
    MetalOffload32MixedLUFactorization()

A mixed precision Metal GPU-accelerated LU factorization that converts matrices to Float32
before offloading to Metal GPU for factorization, then converts back for the solve.
This can provide speedups on Apple Silicon when reduced precision is acceptable.

## Performance Notes
- Converts Float64 matrices to Float32 for GPU factorization
- Can be significantly faster for large matrices where memory bandwidth is limiting
- Particularly effective on Apple Silicon Macs with unified memory architecture
- May have reduced accuracy compared to full precision methods

## Requirements
Using this solver requires that Metal.jl is loaded: `using Metal`

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
    function BLISLUFactorization(; throwerror = true)
        ext = Base.get_extension(@__MODULE__, :LinearSolveBLISExt)
        if ext === nothing && throwerror
            error("BLISLUFactorization requires that the BLIS extension is loaded and blis_jll is available")
        else
            return new()
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
    RF32MixedLUFactorization{P, T}(; pivot = Val(true), thread = Val(true))

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
