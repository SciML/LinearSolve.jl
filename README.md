# LinearSolvers

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://EdelmanJonathan.github.io/LinearSolvers.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://EdelmanJonathan.github.io/LinearSolvers.jl/dev)
[![Build Status](https://github.com/EdelmanJonathan/LinearSolvers.jl/workflows/CI/badge.svg)](https://github.com/EdelmanJonathan/LinearSolvers.jl/actions)
[![Coverage](https://codecov.io/gh/EdelmanJonathan/LinearSolvers.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/EdelmanJonathan/LinearSolvers.jl)

Fast implementations of linear solving algorithms in Julia that satisfy the SciML
common interface. LinearSolve.jl makes it easy to define high level algorithms
which allow for swapping out the linear solver that is used while maintaining
maximum efficiency.

For information on using the package,
[see the stable documentation](https://nonlinearsolve.sciml.ai/stable/). Use the
[in-development documentation](https://nonlinearsolve.sciml.ai/dev/) for the version of
the documentation which contains the unreleased features.

## High Level Examples

```julia
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

linsolve = LinearSolve.set_b(linsolve,b2)
sol2 = solve(linsolve)

sol2.u
#=
4-element Vector{Float64}:
  1.0321556637762768
  0.49724400693338083
 -1.1696540870182406
 -0.4998342686003478
=#

linsolve = LinearSolve.set_b(linsolve,b2)
sol2 = solve(linsolve,IterativeSolversJL_GMRES()) # Switch to GMRES
sol2.u
#=
4-element Vector{Float64}:
  1.0321556637762768
  0.49724400693338083
 -1.1696540870182406
 -0.4998342686003478
=#

A2 = rand(n,n)
linsolve = LinearSolve.set_A(linsolve,A2)
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
