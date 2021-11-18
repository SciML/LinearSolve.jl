#
mutable struct DefaultLinSolve <: SciMLLinearSolveAlgorithm
    iterable
end

DefaultLinSolve() = DefaultLinSolve(nothing, nothing, nothing)

function SciMLBase.solve(cache::LinearCache,
                         alg::DefaultLinSolve,
                         args...;kwargs...)
    @unpack iterable = alg
    @unpack A, b, u, cacheval = cache

    return A \ u
end
