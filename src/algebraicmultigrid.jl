"""
    AlgebraicMultigridJL(args...; kwargs...)

A wrapper for AlgebraicMultigrid.jl solvers.

# Arguments

- `args...`: Arguments passed to the AlgebraicMultigrid solver constructor.
  For example, `AlgebraicMultigrid.ruge_stuben` or `AlgebraicMultigrid.smoothed_aggregation`.
  If no argument is passed, it defaults to `AlgebraicMultigrid.ruge_stuben`.

# Keyword Arguments

- `kwargs...`: Keyword arguments passed to the AlgebraigMultigrid solver constructor.

# Examples

```julia
using LinearSolve, AlgebraicMultigrid
# Use Ruge-Stuben (default)
alg = AlgebraicMultigridJL()

# Use Smoothed Aggregation
alg = AlgebraicMultigridJL(AlgebraicMultigrid.smoothed_aggregation)

# With keywords
alg = AlgebraicMultigridJL(AlgebraicMultigrid.ruge_stuben, presmoother=AlgebraicMultigrid.Jacobi(1.0))
```

See [AlgebraicMultigrid.jl](https://github.com/JuliaLinearAlgebra/AlgebraicMultigrid.jl) for more details on the available solvers and options.
"""
struct AlgebraicMultigridJL{A, K} <: SciMLLinearSolveAlgorithm
    args::A
    kwargs::K
end

function AlgebraicMultigridJL(args...; kwargs...)
    return AlgebraicMultigridJL(args, kwargs)
end

needs_concrete_A(::AlgebraicMultigridJL) = true
