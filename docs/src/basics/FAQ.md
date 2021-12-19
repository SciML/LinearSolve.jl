# Frequently Asked Questions

Ask more questions.

## How do I use IterativeSolvers solvers with a weighted tolerance vector?

IterativeSolvers.jl computes the norm after the application of the left precondtioner
`Pl`. Thus in order to use a vector tolerance `weights`, one can mathematically
hack the system via the following formulation:

```julia
using LinearSolve, LinearAlgebra
Pl = LinearSolve.InvPreconditioner(Diagonal(weights))
Pr = Diagonal(weights)

A = rand(n,n)
b = rand(n)

prob = LinearProblem(A,b)
sol = solve(prob,IterativeSolvers_GMRES(),Pl=Pl,Pr=Pr)
```

If you want to use a "real" preconditioner under the norm `weights`, then one
can use `ComposePreconditioner` to apply the preconditioner after the application
of the weights like as follows:

```julia
using LinearSolve, LinearAlgebra
Pl = ComposePreconitioner(LinearSolve.InvPreconditioner(Diagonal(weights),realprec))
Pr = Diagonal(weights)

A = rand(n,n)
b = rand(n)

prob = LinearProblem(A,b)
sol = solve(prob,IterativeSolvers_GMRES(),Pl=Pl,Pr=Pr)
```
