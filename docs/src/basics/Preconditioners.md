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
s = rand(n)

# Pr applies 1 ./ s .* vec
Pr = LinearSolve.DiagonalPreconditioner(s)
# Pl applies s .* vec
Pl = LinearSolve.DiagonalPreconditioner(s)

A = rand(n,n)
b = rand(n)

prob = LinearProblem(A,b)
sol = solve(prob,IterativeSolvers_GMRES(),Pl=Pl,Pr=Pr)
```

## Pre-Defined Preconditioners

To simplify the usage of preconditioners, LinearSolve.jl comes with many standard
preconditioners written to match the required interface.

- `DiagonalPreconditioner(s::Union{Number,AbstractVector})`: the diagonal
  preconditioner, defined as a diagonal matrix `Diagonal(s)`.

## Preconditioner Interface

To define a new preconditioner you define a Julia type which satisfies the
following interface:

### General

- `Base.eltype(::Preconditioner)`
- `Base.adjoint(::Preconditioner)`
- `Base.inv(::Preconditioner)` (Optional?)

### Required for Right Preconditioners

- `Base.\(::Preconditioner,::AbstractVector)`
- `LinearAlgebra.ldiv!(::AbstractVector,::Preconditioner,::AbstractVector)`

### Required for Left Preconditioners

- `Base.*(::Preconditioner,::AbstractVector)`
- `LinearAlgebra.mul!(::AbstractVector,::Preconditioner,::AbstractVector)`
