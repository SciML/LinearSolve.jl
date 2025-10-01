"""
    LinearSolveFunction{F} <: AbstractSolveFunction

A flexible wrapper that allows using custom functions as linear solver algorithms.
This provides a way to integrate user-defined solving strategies into the LinearSolve.jl
framework while maintaining compatibility with the caching and interface systems.

## Fields
- `solve_func::F`: A callable that implements the custom linear solving logic

## Function Signature

The wrapped function should have the signature:
```julia
solve_func(A, b, u, p, isfresh, Pl, Pr, cacheval; kwargs...)
```

## Arguments to wrapped function
- `A`: The matrix operator of the linear system  
- `b`: The right-hand side vector
- `u`: Pre-allocated solution vector (can be used as working space)
- `p`: Parameters passed to the solver
- `isfresh`: Boolean indicating if the matrix `A` has changed since last solve
- `Pl`: Left preconditioner operator
- `Pr`: Right preconditioner operator  
- `cacheval`: Algorithm-specific cache storage
- `kwargs...`: Additional keyword arguments

## Returns
The wrapped function should return a solution vector.

## Example

```julia
function my_custom_solver(A, b, u, p, isfresh, Pl, Pr, cacheval; kwargs...)
    # Custom solving logic here
    return A \\ b  # Simple example
end

alg = LinearSolveFunction(my_custom_solver)
sol = solve(prob, alg)
```
"""
struct LinearSolveFunction{F} <: AbstractSolveFunction
    solve_func::F
end

function SciMLBase.solve!(cache::LinearCache, alg::LinearSolveFunction,
        args...; kwargs...)
    @unpack A, b, u, p, isfresh, Pl, Pr, cacheval = cache
    @unpack solve_func = alg

    u = solve_func(A, b, u, p, isfresh, Pl, Pr, cacheval; kwargs...)
    return SciMLBase.build_linear_solution(alg, u, nothing, cache)
end

"""
    DirectLdiv! <: AbstractSolveFunction

A simple linear solver that directly applies the left-division operator (`\\`) 
to solve the linear system. This algorithm calls `ldiv!(u, A, b)` which computes
`u = A \\ b` in-place.

## Usage

```julia
alg = DirectLdiv!()
sol = solve(prob, alg)
```

## Notes

- This is essentially a direct wrapper around Julia's built-in `ldiv!` function
- Suitable for cases where the matrix `A` has a natural inverse or factorization
- Performance depends on the specific matrix type and its `ldiv!` implementation
- No preconditioners or advanced numerical techniques are applied
- Best used for small to medium problems or when `A` has special structure
"""
struct DirectLdiv! <: AbstractSolveFunction end

function SciMLBase.solve!(cache::LinearCache, alg::DirectLdiv!, args...; kwargs...)
    @unpack A, b, u = cache
    ldiv!(u, A, b)

    return SciMLBase.build_linear_solution(alg, u, nothing, cache)
end
