# Getting Started with Solving Eigenvalue Problems in Julia

An eigenvalue problem $$Av = \lambda v$$ (or, generalized, $$Av = \lambda Bv$$) is
specified by defining an `EigenvalueProblem`. As with `LinearProblem`, the same
`EigenvalueProblem` can be solved with different algorithms just by swapping out the
`alg` argument to `solve`.

```@example eigsys1
import LinearSolve as LS
import LinearAlgebra

A = LinearAlgebra.Diagonal([1.0, 2.0, 3.0, 4.0])
prob = LS.EigenvalueProblem(A)
sol = LS.solve(prob)
sol.u
```

`sol.u` holds the eigenvalues and `sol.vectors` holds the corresponding eigenvectors as
the columns of a matrix, satisfying `A * sol.vectors ≈ sol.vectors * Diagonal(sol.u)`.
With no algorithm specified, `LS.DenseEigen()` is used, which wraps
`LinearAlgebra.eigen` and then selects the requested eigenpairs out of the full
decomposition.

## Requesting a Subset of Eigenpairs

For large problems, computing every eigenpair is wasteful when only a handful are
needed. `num_eigenpairs` controls how many eigenpairs are returned, and `eigentarget`
(an [`EigenvalueTarget`](@ref)) controls which part of the spectrum they come from:

```@example eigsys1
prob = LS.EigenvalueProblem(
    A; num_eigenpairs = 2, eigentarget = LS.EigenvalueTarget.SmallestMagnitude
)
sol = LS.solve(prob)
sol.u
```

Alternatively, `shift` requests the eigenvalues nearest a given value
(shift-and-invert):

```@example eigsys1
prob = LS.EigenvalueProblem(A; num_eigenpairs = 2, shift = 2.2)
sol = LS.solve(prob)
sol.u
```

## Generalized Eigenvalue Problems

Passing a second operator `B` solves the generalized problem $$Av = \lambda Bv$$
instead:

```@example eigsys1
B = LinearAlgebra.Diagonal(fill(2.0, 4))
prob = LS.EigenvalueProblem(A, B; num_eigenpairs = 2, eigentarget = LS.EigenvalueTarget.LargestRealPart)
sol = LS.solve(prob)
sol.u
```

## Switching to an Iterative Backend

For large sparse or structured matrices, an iterative Krylov-based backend is often
preferable to the dense decomposition. Just as with `LinearProblem`, changing the
solver is a matter of swapping out the algorithm passed to `solve`:

```julia
using KrylovKit
sol = LS.solve(LS.EigenvalueProblem(A; num_eigenpairs = 2), LS.KrylovKitEigen())
sol.u
```

For more information on the available eigenvalue backends and when to choose each one,
see [the eigenvalue solvers page](@ref eigenvaluesolvers).
