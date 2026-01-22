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
