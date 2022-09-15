# Solving Linear Systems in Julia

A linear system $$Au=b$$ is specified by defining an `AbstractMatrix` `A`, or
by providing a matrix-free operator for performing `A*x` operations via the
function `A(u,p,t)` out-of-place and `A(du,u,p,t)` for in-place. For the sake
of simplicity, this tutorial will only showcase concrete matrices.

The following defines a matrix and a `LinearProblem` which is subsequently solved
by the default linear solver.

```@example linsys1
using LinearSolve

A = rand(4,4)
b = rand(4)
prob = LinearProblem(A, b)
sol = solve(prob)
sol.u
```

Note that `solve(prob)` is equivalent to `solve(prob,nothing)` where `nothing`
denotes the choice of the default linear solver. This is equivalent to the
Julia built-in `A\b`, where the solution is recovered via `sol.u`. The power
of this package comes into play when changing the algorithms. For example,
[Krylov.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl)
has some nice methods like GMRES which can be faster in some cases. With
LinearSolve.jl, there is one interface and changing linear solvers is simply
the switch of the algorithm choice:

```@example linsys1
sol = solve(prob,KrylovJL_GMRES())
sol.u
```

Thus a package which uses LinearSolve.jl simply needs to allow the user to
pass in an algorithm struct and all wrapped linear solvers are immediately
available as tweaks to the general algorithm.
