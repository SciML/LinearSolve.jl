# Preconditioners

Many linear solvers can be accelerated by using what is known as a **preconditioner**,
an approximation to the matrix inverse action which is cheap to evaluate. These
can improve the numerical conditioning of the solver process and in turn improve
the performance. LinearSolve.jl provides an interface for the definition of
preconditioners which works with the wrapped packages.

## Using Preconditioners

### Mathematical Definition

Preconditioners are specified in the keyword arguments of `init` or `solve`. The
right preconditioner, `Pr` transforms the linear system ``Au = b`` into the
form:

```math
AP_r^{-1}(Pu) = AP_r^{-1}y = b
```

to add the solving step ``P_r u = y``. The left preconditioner, `Pl`, transforms
the linear system into the form:

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

The following preconditioners are tested to match the interface of LinearSolve.jl.

- `ComposePreconditioner(prec1,prec2)`: composes the preconditioners to apply
  `prec1` before `prec2`.
- `InvPreconditioner(prec)`: inverts `mul!` and `ldiv!` in a preconditioner
  definition as a lazy inverse.
- `LinearAlgera.Diagonal(s::Union{Number,AbstractVector})`: the lazy Diagonal
  matrix type of Base.LinearAlgebra. Used for efficient construction of a
  diagonal preconditioner.
- Other `Base.LinearAlgera` types: all define the full Preconditioner interface.
- [IncompleteLU.ilu](https://github.com/haampie/IncompleteLU.jl): an implementation
  of the incomplete LU-factorization preconditioner. This requires `A` as a
  `SparseMatrixCSC`.
