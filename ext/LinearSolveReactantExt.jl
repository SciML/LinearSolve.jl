module LinearSolveReactantExt

using LinearAlgebra: UnitLowerTriangular, UnitUpperTriangular, ldiv!
using LinearSolve: LinearSolve, DirectLdiv!, OperatorAssumptions
using Reactant: Reactant
using SciMLBase: SciMLBase, LinearProblem, LinearSolution, ReturnCode

Reactant._parent_type(::Type{T}) where {T <: LinearSolution} = T

algorithm_name(alg) = isnothing(alg) ? "the default algorithm" : nameof(typeof(alg))

function solve_route(prob, inputs, args, kwargs)
    alg = isempty(args) ? nothing : first(args)
    if prob.A isa Union{UnitLowerTriangular, UnitUpperTriangular}
        # Promotion checks for real, at-least-single precision without unwrapping a traced scalar.
        T = eltype(prob.A)
        if alg isa DirectLdiv! && length(args) == 1 && isempty(kwargs) &&
                isempty(prob.kwargs) && prob.u0 === nothing &&
                parent(prob.A) isa DenseMatrix && prob.b isa DenseVecOrMat &&
                T === eltype(prob.b) && T === promote_type(real(T), Float32)
            return :triangular
        end
        throw(
            ArgumentError(
                "Reactant cannot lower $(algorithm_name(alg)) for a unit-triangular matrix with these operands or options. " *
                    "Use DirectLdiv!() with matching Float32 or Float64 dense traced arrays, " *
                    "a vector or matrix right-hand side, and default solve options."
            )
        )
    end
    if !applicable(Reactant.Ops.julia_callback, identity, (), inputs...)
        throw(
            ArgumentError(
                "Reactant cannot lower $(algorithm_name(alg)) with A::$(typeof(prob.A)) and b::$(typeof(prob.b)). " *
                    "The Julia callback requires dense traced buffers for A, b, and any initial guess."
            )
        )
    end
    return :callback
end

function reactant_solve(prob::LinearProblem, args...; kwargs...)
    inputs = prob.u0 === nothing ? (prob.A, prob.b) : (prob.A, prob.b, prob.u0)
    if solve_route(prob, inputs, args, kwargs) === :triangular
        u = ldiv!(similar(prob.b), prob.A, prob.b)
        return SciMLBase.build_linear_solution(
            only(args), u, nothing, nothing; retcode = ReturnCode.Success
        )
    end
    output_size = prob.u0 === nothing ? (size(prob.A, 2), Base.tail(size(prob.b))...) :
        size(prob.u0)
    output_eltype = prob.u0 === nothing ? eltype(prob.b) : eltype(prob.u0)
    callback = let prob = prob, args = args, kwargs = kwargs
        function (output, A, b, u0...)
            runtime_prob = isempty(u0) ? SciMLBase.remake(prob; A, b) :
                SciMLBase.remake(prob; A, b, u0 = only(u0))
            sol = SciMLBase.solve(runtime_prob, args...; kwargs...)
            copyto!(output, sol.u)
            return nothing
        end
    end
    u = Reactant.Ops.julia_callback(
        callback, ((output_eltype, output_size),), inputs...
    )
    alg = if isempty(args) || first(args) === nothing
        assump = get(
            kwargs, :assump,
            OperatorAssumptions(size(prob.A, 1) == size(prob.A, 2))
        )
        LinearSolve.defaultalg(prob.A, prob.b, assump)
    else
        first(args)
    end
    return SciMLBase.build_linear_solution(
        alg, u, nothing, nothing; retcode = ReturnCode.Success
    )
end

Reactant.@reactant_overlay function SciMLBase.solve(
        prob::LinearProblem, args...; kwargs...
    )
    return reactant_solve(prob, args...; kwargs...)
end

end
