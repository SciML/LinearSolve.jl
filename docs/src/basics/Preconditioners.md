# [Preconditioners](@id prec)

Many linear solvers can be accelerated by using what is known as a **preconditioner**,
an approximation to the matrix inverse action which is cheap to evaluate. These
can improve the numerical conditioning of the solver process and in turn improve
the performance. LinearSolve.jl provides an interface for the definition of
preconditioners which works with the wrapped iterative solver packages.

## Using Preconditioners

### Mathematical Definition

A right preconditioner, ``P_r`` transforms the linear system ``Au = b`` into the form:

```math
AP_r^{-1}(P_r u) = AP_r^{-1}y = b
```

which is solved for ``y``, and then ``P_r u = y`` is solved for ``u``. The left
preconditioner, ``P_l``, transforms the linear system into the form:

```math
P_l^{-1}Au = P_l^{-1}b
```

A two-sided preconditioned system is of the form:

```math
P_l^{-1}A P_r^{-1} (P_r u) = P_l^{-1}b
```

### Specifying  Preconditioners

One way to specify preconditioners uses the `Pl` and `Pr`  keyword arguments to `init` or `solve`: `Pl` for left
and `Pr` for right preconditioner, respectively. By default, if no preconditioner is given, the preconditioner is assumed to be
the identity ``I``.

In the following, we will use a left sided diagonal (Jacobi) preconditioner.

```@example precon1
using LinearSolve, LinearAlgebra
n = 4

A = rand(n, n)
b = rand(n)

Pl = Diagonal(A)

prob = LinearProblem(A, b)
sol = solve(prob, KrylovJL_GMRES(), Pl = Pl)
sol.u
```

Alternatively, preconditioners can be specified via the  `precs`  argument to the constructor of
an iterative solver specification. This argument shall deliver a factory method mapping `A` and a
parameter `p` to a tuple `(Pl,Pr)` consisting a left and a right preconditioner.

```@example precon2
using LinearSolve, LinearAlgebra
n = 4

A = rand(n, n)
b = rand(n)

prob = LinearProblem(A, b)
sol = solve(prob, KrylovJL_GMRES(precs = (A, p) -> (Diagonal(A), I)))
sol.u
```

This approach has the advantage that the specification of the preconditioner is possible without
the knowledge of a concrete matrix `A`. It also allows to specify the preconditioner via a callable object
and to  pass parameters to the constructor of the preconditioner instances. The example below also shows how
to reuse the preconditioner once constructed for the subsequent solution of a modified problem.

```@example precon3
using LinearSolve, LinearAlgebra

Base.@kwdef struct WeightedDiagonalPreconBuilder
    w::Float64
end

(builder::WeightedDiagonalPreconBuilder)(A, p) = (builder.w * Diagonal(A), I)

n = 4
A = n * I - rand(n, n)
b = rand(n)

prob = LinearProblem(A, b)
sol = solve(prob, KrylovJL_GMRES(precs = WeightedDiagonalPreconBuilder(w = 0.9)))
sol.u

B = A .+ 0.1
cache = sol.cache
reinit!(cache, A = B, reuse_precs = true)
sol = solve!(cache, KrylovJL_GMRES(precs = WeightedDiagonalPreconBuilder(w = 0.9)))
sol.u
```

## Preconditioner Interface

To define a new preconditioner you define a Julia type which satisfies the
following interface:

  - `Base.eltype(::Preconditioner)` (Required only for Krylov.jl)
  - `LinearAlgebra.ldiv!(::AbstractVector,::Preconditioner,::AbstractVector)` and
    `LinearAlgebra.ldiv!(::Preconditioner,::AbstractVector)`

## Curated List of Pre-Defined Preconditioners

The following preconditioners match the interface of LinearSolve.jl.

  - `LinearSolve.ComposePreconditioner(prec1,prec2)`: composes the preconditioners to apply
    `prec1` before `prec2`.

  - `LinearSolve.InvPreconditioner(prec)`: inverts `mul!` and `ldiv!` in a preconditioner
    definition as a lazy inverse.
  - `LinearAlgera.Diagonal(s::Union{Number,AbstractVector})`: the lazy Diagonal
    matrix type of Base.LinearAlgebra. Used for efficient construction of a
    diagonal preconditioner.
  - Other `Base.LinearAlgera` types: all define the full Preconditioner interface.
  - [IncompleteLU.ilu](https://github.com/haampie/IncompleteLU.jl): an implementation
    of the incomplete LU-factorization preconditioner. This requires `A` as a
    `SparseMatrixCSC`.
  - [Preconditioners.CholeskyPreconditioner(A, i)](https://github.com/JuliaLinearAlgebra/Preconditioners.jl):
    An incomplete Cholesky preconditioner with cut-off level `i`. Requires `A` as
    a `AbstractMatrix` and positive semi-definite.
  - [AlgebraicMultigrid](https://github.com/JuliaLinearAlgebra/AlgebraicMultigrid.jl):
    Implementations of the algebraic multigrid method. Must be converted to a
    preconditioner via `AlgebraicMultigrid.aspreconditioner(AlgebraicMultigrid.precmethod(A))`.
    Requires `A` as a `AbstractMatrix`. Provides the following methods:
    
      + `AlgebraicMultigrid.ruge_stuben(A)`
      + `AlgebraicMultigrid.smoothed_aggregation(A)`
  - [PyAMG](https://github.com/cortner/PyAMG.jl):
    Implementations of the algebraic multigrid method. Must be converted to a
    preconditioner via `PyAMG.aspreconditioner(PyAMG.precmethod(A))`.
    Requires `A` as a `AbstractMatrix`. Provides the following methods:
    
      + `PyAMG.RugeStubenSolver(A)`
      + `PyAMG.SmoothedAggregationSolver(A)`
  - [ILUZero.ILU0Precon(A::SparseMatrixCSC{T,N}, b_type = T)](https://github.com/mcovalt/ILUZero.jl):
    An incomplete LU implementation. Requires `A` as a `SparseMatrixCSC`.
  - [LimitedLDLFactorizations.lldl](https://github.com/JuliaSmoothOptimizers/LimitedLDLFactorizations.jl):
    A limited-memory LDLᵀ factorization for symmetric matrices. Requires `A` as a
    `SparseMatrixCSC`. Applying `F = lldl(A); F.D .= abs.(F.D)` before usage as a preconditioner
    makes the preconditioner symmetric positive definite and thus is required for Krylov methods which
    are specialized for symmetric linear systems.
  - [RandomizedPreconditioners.NystromPreconditioner](https://github.com/tjdiamandis/RandomizedPreconditioners.jl)
    A randomized sketching method for positive semidefinite matrices `A`. Builds a preconditioner ``P ≈ A + μ*I``
    for the system ``(A + μ*I)x = b``.
  - [HYPRE.jl](https://github.com/fredrikekre/HYPRE.jl) A set of solvers with
    preconditioners which supports distributed computing via MPI. These can be
    written using the LinearSolve.jl interface choosing algorithms like `HYPRE.ILU`
    and `HYPRE.BoomerAMG`.
  - [KrylovPreconditioners.jl](https://github.com/JuliaSmoothOptimizers/KrylovPreconditioners.jl/): Provides GPU-ready
    preconditioners via KernelAbstractions.jl. At the time of writing the package provides the following methods:
    
      + Incomplete Cholesky decomposition `KrylovPreconditioners.kp_ic0(A)`
      + Incomplete LU decomposition `KrylovPreconditioners.kp_ilu0(A)`
      + Block Jacobi `KrylovPreconditioners.BlockJacobiPreconditioner(A, nblocks, device)`
