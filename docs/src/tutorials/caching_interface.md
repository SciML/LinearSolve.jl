# Linear Solve with Caching Interface

In many cases one may want to cache information that is reused between different
linear solves. For example, if one is going to perform:

```julia
A\b1
A\b2
```

then it would be more efficient to LU-factorize one time and reuse the factorization:

```julia
lu!(A)
A\b1
A\b2
```

LinearSolve.jl's caching interface automates this process to use the most efficient
means of solving and resolving linear systems. To do this with LinearSolve.jl,
you simply `init` a cache, `solve`, replace `b`, and solve again. This looks like:

```julia
using LinearSolve

n = 4
A = rand(n,n)
b1 = rand(n); b2 = rand(n)
prob = LinearProblem(A, b1)

linsolve = init(prob)
sol1 = solve(linsolve)

sol1.u
#=
4-element Vector{Float64}:
 -0.9247817429364165
 -0.0972021708185121
  0.6839050402960025
  1.8385599677530706
=#

linsolve = LinearSolve.set_b(sol1.cache,b2)
sol2 = solve(linsolve)

sol2.u
#=
4-element Vector{Float64}:
  1.0321556637762768
  0.49724400693338083
 -1.1696540870182406
 -0.4998342686003478
=#
```

Then refactorization will occur when a new `A` is given:

```julia
A2 = rand(n,n)
linsolve = LinearSolve.set_A(sol2.cache,A2)
sol3 = solve(linsolve)

sol3.u
#=
4-element Vector{Float64}:
 -6.793605395935224
  2.8673042300837466
  1.1665136934977371
 -0.4097250749016653
=#
```

The factorization occurs on the first solve, and it stores the factorization in
the cache. You can retrieve this cache via `sol.cache`, which is the same object
as the `init` but updated to know not to re-solve the factorization.

The advantage of course with using LinearSolve.jl in this form is that it is
efficient while being agnostic to the linear solver. One can easily swap in
iterative solvers, sparse solvers, etc. and it will do all of the tricks like
caching symbolic factorizations if the sparsity pattern is unchanged.
