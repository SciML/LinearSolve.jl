# Linear Solve with Caching Interface

Often, one may want to cache information that is reused between different
linear solves. For example, if one is going to perform:

```julia
A \ b1
A \ b2
```

then it would be more efficient to LU-factorize one time and reuse the factorization:

```julia
lu!(A)
A \ b1
A \ b2
```

LinearSolve.jl's caching interface automates this process to use the most efficient
means of solving and resolving linear systems. To do this with LinearSolve.jl,
you simply `init` a cache, `solve`, replace `b`, and solve again. This looks like:

```@example linsys2
using LinearSolve

n = 4
A = rand(n, n)
b1 = rand(n);
b2 = rand(n);
prob = LinearProblem(A, b1)

linsolve = init(prob)
sol1 = solve(linsolve)
```

```@example linsys2
linsolve.b = b2
sol2 = solve(linsolve)

sol2.u
```

Then refactorization will occur when a new `A` is given:

```@example linsys2
A2 = rand(n, n)
linsolve.A = A2
sol3 = solve(linsolve)

sol3.u
```

The factorization occurs on the first solve, and it stores the factorization in
the cache. You can retrieve this cache via `sol.cache`, which is the same object
as the `init`, but updated to know not to re-solve the factorization.

The advantage of course with using LinearSolve.jl in this form is that it is
efficient while being agnostic to the linear solver. One can easily swap in
iterative solvers, sparse solvers, etc. and it will do all the tricks like
caching the symbolic factorization if the sparsity pattern is unchanged.
