# Solving Linear Systems in Julia

A linear system $$Au=b$$ is specified by defining an `AbstractMatrix` `A`, or
by providing a matrix-free operator for performing `A*x` operations via the
function `A(u,p,t)` out-of-place and `A(du,u,p,t)` for in-place. For the sake
of simplicity, this tutorial will only showcase concrete matrices.

The following defines a matrix and a `LinearProblem` which is subsequently solved
by the default linear solver.

```julia
using LinearSolve

A = rand(4,4)
b = rand(4)
prob = LinearProblem(A, b)
sol = solve(prob)
sol.u

#=
4-element Vector{Float64}:
  0.3784870087078603
  0.07275749718047864
  0.6612816064734302
 -0.10598367531463938
=#
```

Note that `solve(prob)` is equivalent to `solve(prob,nothing)` where `nothing`
denotes the choice of the default linear solver. This is equivalent to the
Julia built-in `A\b`, where the solution is recovered via `sol.u`. The power
of this package comes into play when changing the algorithms. For example,
[IterativeSolvers.jl](https://github.com/JuliaLinearAlgebra/IterativeSolvers.jl)
has some nice methods like GMRES which can be faster in some cases. With
LinearSolve.jl, there is one interface and changing linear solvers is simply
the switch of the algorithm choice:

```julia
sol = solve(prob,IterativeSolversJL_GMRES())

#=
u: 4-element Vector{Float64}:
  0.37848700870786
  0.07275749718047908
  0.6612816064734302
 -0.10598367531463923
=#
```

Thus a package which uses LinearSolve.jl simply needs to allow the user to
pass in an algorithm struct and all wrapped linear solvers are immediately
available as tweaks to the general algorithm.
