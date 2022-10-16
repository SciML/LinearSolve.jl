# LinearSolve.jl

[![Join the chat at https://julialang.zulipchat.com #sciml-bridged](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
[![Global Docs](https://img.shields.io/badge/docs-SciML-blue.svg)](https://docs.sciml.ai/LinearSolve/stable/)

[![codecov](https://codecov.io/gh/SciML/LinearSolve.jl/branch/main/graph/badge.svg)](https://app.codecov.io/gh/SciML/LinearSolve.jl)
[![Build Status](https://github.com/SciML/LinearSolvers.jl/workflows/CI/badge.svg)](https://github.com/SciML/LinearSolvers.jl/actions?query=workflow%3ACI)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

Fast implementations of linear solving algorithms in Julia that satisfy the SciML
common interface. LinearSolve.jl makes it easy to define high level algorithms
which allow for swapping out the linear solver that is used while maintaining
maximum efficiency. Specifically, LinearSolve.jl includes:

- Fast pure Julia LU factorizations which outperform standard BLAS
- KLU for faster sparse LU factorization on unstructured matrices
- UMFPACK for faster sparse LU factorization on matrices with some repeated structure
- MKLPardiso wrappers for handling many sparse matrices faster than SuiteSparse (KLU, UMFPACK) methods
- GPU-offloading for large dense matrices
- Wrappers to all of the Krylov implementations (Krylov.jl, IterativeSolvers.jl, KrylovKit.jl) for easy
  testing of all of them. LinearSolve.jl handles the API differences, especially with the preconditioner
  definitions
- A polyalgorithm that smartly chooses between these methods
- A caching interface which automates caching of symbolic factorizations and numerical factorizations
  as optimally as possible

For information on using the package,
[see the stable documentation](https://docs.sciml.ai/LinearSolve/stable/). Use the
[in-development documentation](https://docs.sciml.ai/LinearSolve/dev/) for the version of
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
