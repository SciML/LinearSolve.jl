module LinearSolveMooncakeExt

using Mooncake
using Mooncake: @from_chainrules, MinimalCtx, ReverseMode, NoRData, increment!!
using LinearSolve: LinearSolve, SciMLLinearSolveAlgorithm, init, solve!, LinearProblem,
                   LinearCache, AbstractKrylovSubspaceMethod, DefaultLinearSolver,
                   defaultalg_adjoint_eval, solve
using LinearSolve.LinearAlgebra
using SciMLBase

@from_chainrules MinimalCtx Tuple{typeof(SciMLBase.solve), LinearProblem, Nothing} true ReverseMode
@from_chainrules MinimalCtx Tuple{
    typeof(SciMLBase.solve), LinearProblem, SciMLLinearSolveAlgorithm} true ReverseMode
@from_chainrules MinimalCtx Tuple{
    Type{<:LinearProblem}, AbstractMatrix, AbstractVector, SciMLBase.NullParameters} true ReverseMode

function Mooncake.increment_and_get_rdata!(f, r::NoRData, t::LinearProblem)
    f.data.A .+= t.A
    f.data.b .+= t.b

    return NoRData()
end

function Mooncake.to_cr_tangent(x::Mooncake.PossiblyUninitTangent{T}) where {T}
    if Mooncake.is_init(x)
        return Mooncake.to_cr_tangent(x.tangent)
    else
        error("Trying to convert uninitialized tangent to ChainRules tangent.")
    end
end

end