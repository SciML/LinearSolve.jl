module LinearSolveReactantExt

using LinearSolve: LinearSolve, OperatorAssumptions
using Reactant: Reactant
using SciMLBase: SciMLBase, LinearProblem, LinearSolution, ReturnCode

Reactant._parent_type(::Type{T}) where {T <: LinearSolution} = T

function reactant_solve(prob::LinearProblem, args...; kwargs...)
    output_size = prob.u0 === nothing ? (size(prob.A, 2), Base.tail(size(prob.b))...) :
        size(prob.u0)
    output_eltype = prob.u0 === nothing ? eltype(prob.b) : eltype(prob.u0)
    inputs = prob.u0 === nothing ? (prob.A, prob.b) : (prob.A, prob.b, prob.u0)
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
