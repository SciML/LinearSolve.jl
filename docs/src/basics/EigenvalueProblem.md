# [Eigenvalue Problems](@id eigenvalue_problem)

`EigenvalueProblem` and `EigenvalueTarget` are SciMLBase types re-exported by
LinearSolve.jl. Their docstrings are maintained in SciMLBase and rendered in the
[SciMLBase problem interface documentation](https://docs.sciml.ai/SciMLBase/stable/interfaces/Problems/);
this page summarizes how LinearSolve.jl uses them. See the
[eigenvalue tutorial](@ref eigenvalue_tutorial) for worked examples and the
[eigenvalue solvers](@ref eigenvaluesolvers) page for the available algorithms.

## Mathematical Specification

The standard problem finds pairs ``(\lambda, v)`` with

```math
A v = \lambda v
```

and, when a second operator `B` is given, the generalized problem

```math
A v = \lambda B v
```

Eigenvectors follow the type of `u0` when supplied, otherwise the dense vector
type matching a row of `A`. Eigenvalues have `eltype(A)` when real and
`Complex{eltype(A)}` when a general real `A` yields conjugate pairs.

## Constructor

```julia
EigenvalueProblem(A, B = nothing, p = NullParameters();
    num_eigenpairs = nothing, eigentarget = EigenvalueTarget.LargestMagnitude,
    shift = nothing, u0 = nothing, kwargs...)
```

  - `num_eigenpairs`: how many eigenpairs to compute; `nothing` requests all of
    them from the dense solver or a solver-chosen default from the iterative
    backends.
  - `eigentarget`: which part of the spectrum to return, an `EigenvalueTarget`
    (below).
  - `shift`: when supplied, the eigenvalues nearest this value are returned
    (shift-and-invert).
  - `u0`: an optional starting vector for the iterative backends.
  - `kwargs`: extra keyword arguments passed on to the solver.

## `EigenvalueTarget`

`EigenvalueTarget` is an [EnumX](https://github.com/fredrikekre/EnumX.jl) enum
selecting the part of the spectrum returned when only a subset of the eigenpairs
is requested:

| Value                                    | Selects                                       |
|:---------------------------------------- |:--------------------------------------------- |
| `EigenvalueTarget.LargestMagnitude`      | largest `abs(λ)` (the default)                |
| `EigenvalueTarget.SmallestMagnitude`     | smallest `abs(λ)`                             |
| `EigenvalueTarget.LargestRealPart`       | largest (most positive) real part             |
| `EigenvalueTarget.SmallestRealPart`      | smallest (most negative) real part            |
| `EigenvalueTarget.LargestImaginaryPart`  | largest (most positive) imaginary part        |
| `EigenvalueTarget.SmallestImaginaryPart` | smallest (most negative) imaginary part       |

Not every backend supports every target directly; the
[eigenvalue solvers](@ref eigenvaluesolvers) page lists each algorithm's
restrictions.

## Example

```julia
using LinearSolve

A = [2.0 1.0; 1.0 3.0]
prob = EigenvalueProblem(A; num_eigenpairs = 1,
    eigentarget = EigenvalueTarget.LargestMagnitude)
sol = solve(prob)
sol.values, sol.vectors
```
