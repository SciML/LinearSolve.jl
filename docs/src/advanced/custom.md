# Passing in a Custom Linear Solver
Julia users are building a wide variety of applications in the SciML ecosystem,
often requiring problem-specific handling of their linear solves. As existing solvers in `LinearSolve.jl` may not
be optimally suited for novel applications, it is essential for the linear solve
interface to be easily extendable by users. To that end, the linear solve algorithm
`LinearSolveFunction()` accepts a user-defined function for handling the solve. A
user can pass in their custom linear solve function, say `my_linsolve`, to
`LinearSolveFunction()`. A contrived example of solving a linear system with a custom solver is below.
```julia
using LinearSolve, LinearAlgebra

function my_linsolve(A,b,u,p,newA,Pl,Pr,solverdata;verbose=true, kwargs...)
    if verbose == true
        println("solving Ax=b")
    end
    u = A \ b
    return u
end

prob = LinearProblem(Diagonal(rand(4)), rand(4))
alg  = LinearSolveFunction(my_linsolve)
sol  = solve(prob, alg)
```
The inputs to the function are as follows:
- `A`, the linear operator
- `b`, the right-hand-side
- `u`, the solution initialized as `zero(b)`,
- `p`, a set of parameters
- `newA`, a `Bool` which is `true` if `A` has been modified since last solve
- `Pl`, left-preconditioner
- `Pr`, right-preconditioner
- `solverdata`, solver cache set to `nothing` if solver hasn't been initialized
- `kwargs`, standard SciML keyword arguments such as `verbose`, `maxiters`, `abstol`, `reltol`

The function `my_linsolve` must accept the above specified arguments, and return
the solution, `u`. As memory for `u` is already allocated, the user may choose
to modify `u` in place as follows:
```julia
function my_linsolve!(A,b,u,p,newA,Pl,Pr,solverdata;verbose=true, kwargs...)
    if verbose == true
        println("solving Ax=b")
    end
    u .= A \ b # in place
    return u
end

alg  = LinearSolveFunction(my_linsolve!)
sol  = solve(prob, alg)
```
