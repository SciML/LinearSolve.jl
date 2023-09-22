module LinearSolveEnzymeExt

using LinearSolve
isdefined(Base, :get_extension) ? (import Enzyme) : (import ..Enzyme)


using Enzyme

using EnzymeCore

# y=inv(A) B
#   dA −= z y^T  
#   dB += z, where  z = inv(A^T) dy
function EnzymeCore.EnzymeRules.augmented_primal(config::EnzymeCore.EnzymeRules.ConfigWidth{1}, func::Const{typeof(LinearSolve.solve)}, ::Type{Duplicated{RT}}, prob::Duplicated{LP}, alg::Const; kwargs...) where {RT, LP <: LinearProblem}
    res = func.val(prob.val, alg.val; kwargs...)
    dres = deepcopy(res)
    dres.u .= 0
    cache = (copy(prob.val.A), res, dres.u)
    return EnzymeCore.EnzymeRules.AugmentedReturn{RT, RT, typeof(cache)}(res, dres, cache)
end

function EnzymeCore.EnzymeRules.reverse(config::EnzymeCore.EnzymeRules.ConfigWidth{1}, func::Const{typeof(LinearSolve.solve)}, ::Type{Duplicated{RT}}, cache, prob::Duplicated{LP}, alg::Const; kwargs...) where {RT, LP <: LinearProblem}
    A, y, dy = cache

    dA = prob.dval.A
    db = prob.dval.b

    invprob = LinearProblem(transpose(A), dy)

    z = func.val(invprob, alg; kwargs...)

    dA .-= z * transpose(y)
    db .+= z
    dy .= 0
    return (nothing, nothing)
end

end