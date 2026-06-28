# Linear Solve with Caching Interface

Often, one may want to cache information that is reused between different
linear solves. For example, if one is going to perform:

```julia
A \ b1
A \ b2
```

then it would be more efficient to LU-factorize one time and reuse the factorization:

```julia
A_lu = LA.lu!(A)
A_lu \ b1
A_lu \ b2
```

LinearSolve.jl's caching interface automates this process to use the most efficient
means of solving and resolving linear systems. To do this with LinearSolve.jl,
you simply `init` a cache, `solve`, replace `b`, and solve again. This looks like:

```@example linsys2
import LinearSolve as LS
import LinearAlgebra as LA

n = 4
A = rand(n, n)
b1 = rand(n);
b2 = rand(n);
prob = LS.LinearProblem(A, b1)

linsolve = LS.init(prob)
sol1 = LS.solve!(linsolve)
```

```@example linsys2
linsolve.b = b2
sol2 = LS.solve!(linsolve)

sol2.u
```

Then refactorization will occur when a new `A` is given:

```@example linsys2
A2 = rand(n, n)
linsolve.A = A2
sol3 = LS.solve!(linsolve)

sol3.u
```

The factorization occurs on the first solve, and it stores the factorization in
the cache. You can retrieve this cache via `sol.cache`, which is the same object
as the `init`, but updated to know not to re-solve the factorization.

The advantage of course with import LinearSolve.jl in this form is that it is
efficient while being agnostic to the linear solver. One can easily swap in
iterative solvers, sparse solvers, etc. and it will do all the tricks like
caching the symbolic factorization if the sparsity pattern is unchanged.

## Reusing the Symbolic Factorization with Sparse Matrices

For sparse factorizations such as `UMFPACKFactorization` and `KLUFactorization`,
a significantly expensive part of a solve is often the *symbolic* factorization: the
analysis of the sparsity pattern that determines the fill-in and elimination
ordering. The subsequent *numeric* factorization, which computes the actual `L`
and `U` factor values, is not cheap but must be recomputed for every change in the
values of `A`. Thus storing the symbolic factorization can lead to some sizable
gains.

A very common case (for example, the Newton steps of a nonlinear solve or the
time steps of an implicit ODE/PDE integrator) is that `A` keeps the **same
sparsity pattern** across solves while only its stored values change. In that
case, the symbolic factorization can be reused and only the numeric
factorization needs to be redone. This is controlled by the `reuse_symbolic`
keyword on the sparse factorization algorithms, which **defaults to `true`** —
so you get this behavior automatically and do not need to set it:

```julia
solver = LS.UMFPACKFactorization()   # reuse_symbolic = true by default
```

When a new `A` with an unchanged pattern is given, LinearSolve.jl reuses the
cached symbolic factorization and performs only the numeric refactorization. By
default it also runs a `check_pattern` pass to confirm the pattern really is
unchanged; if your pattern is guaranteed constant, you can set
`check_pattern = false` to skip that check for a little extra speed. Set
`reuse_symbolic = false` only if the sparsity pattern may change between solves.

### Updating the values of `A`

The caching interface tracks whether the factorization is stale through the
assignment `linsolve.A = A`. **Mutating the stored values of `A` in place does
not, on its own, tell the cache that `A` has changed**, so the next `solve!`
will silently reuse the old factorization and return a stale (wrong) result:

```julia
linsolve.A.nzval .*= 2   # in-place edit: NOT seen by the cache
LS.solve!(linsolve)      # reuses the old factorization -> wrong answer
```

The supported way to signal that the values changed is to assign `A` back into
the cache. Reassigning the *same* (now-mutated) matrix object is fine and still
reuses the symbolic factorization when the pattern is unchanged:

```@example sparsereuse
import LinearSolve as LS
import LinearAlgebra as LA
import SparseArrays

n = 100
A = SparseArrays.sprand(n, n, 0.05) + 10.0 * LA.I
b = rand(n)
prob = LS.LinearProblem(A, b)

linsolve = LS.init(prob, LS.UMFPACKFactorization())
sol1 = LS.solve!(linsolve)

# Update the values of A (same sparsity pattern), then signal the change:
A.nzval .*= 2
linsolve.A = A      # triggers refactorization; symbolic part is reused
sol2 = LS.solve!(linsolve)

sol2.u
```

Because the pattern is unchanged, `sol2` reuses the cached symbolic
factorization and only redoes the (cheaper) numeric factorization.

!!! note

    Internally, the cache uses an `isfresh` flag to mark the factorization as
    stale, and assigning `linsolve.A = A` sets it. If you must edit `A`'s values
    in place and want to avoid touching the matrix field, you can equivalently
    set `linsolve.isfresh = true` directly. Prefer `linsolve.A = A`, as
    `isfresh` is an internal detail. Either way, `reuse_symbolic = true` ensures
    the symbolic factorization is reused as long as the sparsity pattern is the
    same.
