module LinearSolvePartitionedSolversExt

using PartitionedArrays: PSparseMatrix, PVector
using PartitionedSolvers: PartitionedSolvers
using LinearSolve: LinearSolve, PartitionedSolversAlgorithm, LinearCache, LinearProblem,
    init_cacheval, __init, OperatorAssumptions, LinearVerbosity
using SciMLBase: SciMLBase

mutable struct PartitionedSolversCache
    solver::Any
end

function LinearSolve.init_cacheval(
        alg::PartitionedSolversAlgorithm, A, b, u, Pl, Pr, maxiters::Int,
        abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return PartitionedSolversCache(nothing)
end

function validate_partitionedsolvers_problem(A, b, u0)
    A isa PSparseMatrix || throw(
        ArgumentError(
            "PartitionedSolversAlgorithm requires A::PSparseMatrix, got $(typeof(A))"
        )
    )
    b isa PVector || throw(
        ArgumentError("PartitionedSolversAlgorithm requires b::PVector, got $(typeof(b))")
    )
    (u0 === nothing || u0 isa PVector) || throw(
        ArgumentError(
            "PartitionedSolversAlgorithm requires u0::PVector when provided, got $(typeof(u0))"
        )
    )
    return nothing
end

function SciMLBase.init(prob::LinearProblem, alg::PartitionedSolversAlgorithm, args...; kwargs...)
    (; A, b, u0, p) = prob
    validate_partitionedsolvers_problem(A, b, u0)
    u0_ = u0 === nothing ? zero(b) : u0
    prob_ = u0 === u0_ ? prob : LinearProblem(A, b, p; u0 = u0_)
    return __init(prob_, alg, args...; kwargs...)
end

function SciMLBase.solve!(cache::LinearCache, alg::PartitionedSolversAlgorithm, args...; kwargs...)
    throw(
        ArgumentError(
            "PartitionedSolversAlgorithm solve path is not implemented yet; week 12-13 will wire this cache to PartitionedSolvers.solve"
        )
    )
end

end # module LinearSolvePartitionedSolversExt
