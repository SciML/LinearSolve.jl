# [Linear Problems](@id linear_problem)

`LinearProblem` is the SciMLBase problem type for linear systems, re-exported by
LinearSolve.jl. Its docstring is maintained in SciMLBase and rendered in the
[SciMLBase problem interface documentation](https://docs.sciml.ai/SciMLBase/stable/interfaces/Problems/);
this page summarizes how LinearSolve.jl uses it.

## Mathematical Specification

A `LinearProblem` defines the system

```math
Au = b
```

where `A` is either a concrete `AbstractMatrix` (dense, sparse, GPU, ...) or a
matrix-free operator from the
[SciMLOperators](https://docs.sciml.ai/SciMLOperators/stable/) interface, and `b`
is the right-hand side. Matrix-free operators are not compatible with every
solver; a solver's `needs_concrete_A(alg)` reports whether it requires a
materialized matrix.

## Constructors

```julia
LinearProblem(A, b, p = NullParameters(); u0 = nothing, kwargs...)
LinearProblem{isinplace}(A, b, p = NullParameters(); u0 = nothing, kwargs...)
LinearProblem(f::AbstractSciMLOperator, b, p = NullParameters(); u0 = nothing, kwargs...)
```

  - `A`, `b`: the operator and right-hand side.
  - `p`: problem parameters, `NullParameters()` when absent (currently unused by
    the solvers).
  - `u0`: an optional initial guess, used by iterative methods.
  - `isinplace`: whether solvers may mutate `A` and `b`; defaults to `true` for
    an `AbstractMatrix` and follows the operator's own setting for an
    `AbstractSciMLOperator`.
  - `kwargs`: any extra keyword arguments are stored on the problem and passed to
    `solve`; see [Common Solver Options](@ref "Common Solver Options (Keyword Arguments for Solve)").

## Fields

  - `A`: the linear operator.
  - `b`: the right-hand side.
  - `p`: the parameters.
  - `u0`: the initial guess, or `nothing`.
  - `symbolic_interface`: a `SymbolicLinearInterface` when the problem was built
    by a symbolic front end, else `nothing`.
  - `kwargs`: keyword arguments forwarded to the solvers.

## Example

```julia
using LinearSolve

A = rand(4, 4)
b = rand(4)
prob = LinearProblem(A, b)
sol = solve(prob)
sol.u
```
