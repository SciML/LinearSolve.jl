# Frequently Asked Questions

Ask more questions.

## How do I use IterativeSolvers solvers with a weighted tolerance vector?

IterativeSolvers.jl computes the norm after the application of the left precondtioner
`Pl`. Thus in order to use a vector tolerance `weights`, one can mathematically
hack the system via the following formulation:

```julia
using LinearSolve, SciMLOperators
Pr = DiagonalOperator(weights)
Pl = inv(Pr)

A = rand(n,n)
b = rand(n)

prob = LinearProblem(A,b)
sol = solve(prob,IterativeSolvers_GMRES(),Pl=Pl,Pr=Pr)
```
Here, `Base.inv` is overloaded on `SciMLOperators` to store a lazy inverse so no memory is allocated in forming `Pl = inv(Pr)`.

If you want to use a "real" preconditioner under the norm `weights`, then one
can lazily compose preconditinoers via SciMLOperators to apply the preconditioner
after the application of the weights as follows:

```julia
using LinearSolve, SciMLOperators
Pr = DiagonalOperator(weights)
Pl = inv(Pr) * realprec

A = rand(n,n)
b = rand(n)

prob = LinearProblem(A,b)
sol = solve(prob,IterativeSolvers_GMRES(),Pl=Pl,Pr=Pr)
```
As before, `inv(Pr)` is formed lazily, and `Base.*` is overloaded to lazy composition for `SciMLOperators`.
