module LinearSolveEnzymeExt

using LinearSolve
isdefined(Base, :get_extension) ? (import Enzyme) : (import ..Enzyme)


using Enzyme

using EnzymeCore

function EnzymeCore.EnzymeRules.augmented_primal(config, func::Const{typeof(LinearSolve.init)}, ::Type{RT}, prob::EnzymeCore.Annotation{LP}, alg::Const; kwargs...) where {RT, LP <: LinearSolve.LinearProblem}
    res = func.val(prob.val, alg.val; kwargs...)
    dres = if EnzymeRules.width(config) == 1
        func.val(prob.dval, alg.val; kwargs...)
    else
        (func.val(dval, alg.val; kwargs...) for dval in prob.dval)
    end
    return EnzymeCore.EnzymeRules.AugmentedReturn(res, dres, nothing)
end

function EnzymeCore.EnzymeRules.reverse(config, func::Const{typeof(LinearSolve.init)}, ::Type{RT}, cache, prob::EnzymeCore.Annotation{LP}, alg::Const; kwargs...) where {RT, LP <: LinearSolve.LinearProblem}
    return (nothing, nothing)
end

# y=inv(A) B
#   dA −= z y^T  
#   dB += z, where  z = inv(A^T) dy
function EnzymeCore.EnzymeRules.augmented_primal(config, func::Const{typeof(LinearSolve.solve!)}, ::Type{RT}, linsolve::EnzymeCore.Annotation{LP}; kwargs...) where {RT, LP <: LinearSolve.LinearCache}
    res = func.val(linsolve.val; kwargs...)
    dres = if EnzymeRules.width(config) == 1
        deepcopy(res)
    else
        (deepcopy(res) for dval in linsolve.dval)
    end

    if EnzymeRules.width(config) == 1
        dres.u .= 0
    else
        for dr in dres
            dr.u .= 0
        end
    end

    resvals = if EnzymeRules.width(config) == 1
        dres.u
    else
        (dr.u for dr in dres)
    end

    cache = (copy(linsolve.val.A), res, resvals)
    return EnzymeCore.EnzymeRules.AugmentedReturn(res, dres, cache)
end

function EnzymeCore.EnzymeRules.reverse(config, func::Const{typeof(LinearSolve.solve!)}, ::Type{RT}, cache, linsolve::EnzymeCore.Annotation{LP}; kwargs...) where {RT, LP <: LinearSolve.LinearCache}
    A, y, dys = cache

    @assert !(typeof(linsolve) <: Const)
    @assert !(typeof(linsolve) <: Active)

    if EnzymeRules.width(config) == 1
        dys = (dys,)
    end

    dAs = if EnzymeRules.width(config) == 1
        (linsolve.dval.A,)
    else
        (dval.A for dval in linsolve.dval)
    end

    dbs = if EnzymeRules.width(config) == 1
        (linsolve.dval.b,)
    else
        (dval.b for dval in linsolve.dval)
    end

    for (dA, db, dy) in zip(dAs, dbs, dys)
        invprob = LinearSolve.LinearProblem(transpose(A), dy)
        z = solve(invprob;
            abstol = linsolve.val.abstol,
            reltol = linsolve.val.reltol,
            verbose = linsolve.val.verbose) 

        dA .-= z * transpose(y)
        db .+= z
        dy .= eltype(dy)(0)
    end

    return (nothing,)
end

end