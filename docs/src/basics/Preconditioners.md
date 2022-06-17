# [Preconditioners](@id prec)

Many linear solvers can be accelerated by using what is known as a **preconditioner**,
an approximation to the matrix inverse action which is cheap to evaluate. These
can improve the numerical conditioning of the solver process and in turn improve
the performance. LinearSolve.jl provides an interface for the definition of
preconditioners which works with the wrapped packages.

## Using Preconditioners

### Mathematical Definition

Preconditioners are specified in the keyword arguments to `init` or `solve`: `Pl` for left
and `Pr` for right preconditioner, respectively.
The right preconditioner, ``P_r`` transforms the linear system ``Au = b`` into the form:

```math
AP_r^{-1}(P_r u) = AP_r^{-1}y = b
```

which is solved for ``y``, and then ``P_r u = y`` is solved for ``u``. The left
preconditioner, ``P_l``, transforms the linear system into the form:

```math
P_l^{-1}(Au - b) = 0
```

A two-sided preconditioned system is of the form:

```math
P_l A P_r^{-1} (P_r u) = P_l b
```

By default, if no preconditioner is given the preconditioner is assumed to be
the identity ``I``.

### Using Preconditioners

In the following, we will use the `DiagonalPreconditioner` to define a two-sided
preconditioned system which first divides by some random numbers and then
multiplies by the same values. This is commonly used in the case where if, instead
of random, `s` is an approximation to the eigenvalues of a system.

```julia
using LinearSolve, LinearAlgebra
s = rand(n)
Pl = Diagonal(s)

A = rand(n,n)
b = rand(n)

prob = LinearProblem(A,b)
sol = solve(prob,IterativeSolvers_GMRES(),Pl=Pl)
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
- [Preconditioners.CholeskyPreconditioner(A, i)](https://github.com/mohamed82008/Preconditioners.jl):
  An incomplete Cholesky preconditioner with cut-off level `i`. Requires `A` as
  a `AbstractMatrix` and positive semi-definite.
- [AlgebraicMultiGrid](https://github.com/JuliaLinearAlgebra/AlgebraicMultigrid.jl):
  Implementations of the algebraic multigrid method. Must be converted to a
  preconditioner via `AlgebraicMultiGrid.aspreconditioner(AlgebraicMultiGrid.precmethod(A))`.
  Requires `A` as a `AbstractMatrix`. Provides the following methods:
  - `AlgebraicMultiGrid.ruge_stuben(A)`
  - `AlgebraicMultiGrid.smoothed_aggregation(A)`
- [PyAMG](https://github.com/cortner/PyAMG.jl):
  Implementations of the algebraic multigrid method. Must be converted to a
  preconditioner via `PyAMG.aspreconditioner(PyAMG.precmethod(A))`.
  Requires `A` as a `AbstractMatrix`. Provides the following methods:
  - `PyAMG.RugeStubenSolver(A)`
  - `PyAMG.SmoothedAggregationSolver(A)`
- [ILUZero.ILU0Precon(A::SparseMatrixCSC{T,N}, b_type = T)](https://github.com/mcovalt/ILUZero.jl):
  An incomplete LU implementation. Requires `A` as a `SparseMatrixCSC`.
- [LimitedLDLFactorizations.lldl](https://github.com/JuliaSmoothOptimizers/LimitedLDLFactorizations.jl):
  A limited-memory LDLᵀ factorization for symmetric matrices. Requires `A` as a
  `SparseMatrixCSC`. Applying `F = lldl(A); F.D .= abs.(F.D)` before usage as a preconditioner
  makes the preconditioner symmetric postive definite and thus is required for Krylov methods which
  are specialized for symmetric linear systems.
- [RandomizedPreconditioners.NystromPreconditioner](https://github.com/tjdiamandis/RandomizedPreconditioners.jl)
  A randomized sketching method for positive semidefinite matrices `A`. Builds a preconditioner ``P ≈ A + μ*I``
  for the system ``(A + μ*I)x = b``
