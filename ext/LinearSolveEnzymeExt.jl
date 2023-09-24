module LinearSolveEnzymeExt

using LinearSolve
using LinearSolve.LinearAlgebra
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
    d_A = if EnzymeRules.width(config) == 1
        dres.A
    else
        (dval.A for dval in dres)
    end
    d_b = if EnzymeRules.width(config) == 1
        dres.b
    else
        (dval.b for dval in dres)
    end
    return EnzymeCore.EnzymeRules.AugmentedReturn(res, dres, (d_A, d_b))
end

function EnzymeCore.EnzymeRules.reverse(config, func::Const{typeof(LinearSolve.init)}, ::Type{RT}, cache, prob::EnzymeCore.Annotation{LP}, alg::Const; kwargs...) where {RT, LP <: LinearSolve.LinearProblem}
    d_A, d_b = cache

    if EnzymeRules.width(config) == 1
        if d_A !== prob.dval.A
            prob.dval.A .+= d_A
            d_A .= 0
        end
        if d_b !== prob.dval.b
            prob.dval.b .+= d_b
            d_b .= 0
        end
    else
        for i in 1:EnzymeRules.width(config)
            if d_A !== prob.dval.A
                prob.dval.A[i] .+= d_A[i]
                d_A[i] .= 0
            end
            if d_b !== prob.dval.b
                prob.dval.b[i] .+= d_b[i]
                d_b[i] .= 0
            end
        end
    end

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

    cache = (res, resvals)
    return EnzymeCore.EnzymeRules.AugmentedReturn(res, dres, cache)
end

function EnzymeCore.EnzymeRules.reverse(config, func::Const{typeof(LinearSolve.solve!)}, ::Type{RT}, cache, linsolve::EnzymeCore.Annotation{LP}; kwargs...) where {RT, LP <: LinearSolve.LinearCache}
    y, dys = cache
    _linsolve = linsolve.val

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
        z = if _linsolve.cacheval isa Factorization
            _linsolve.cacheval' \ dy
        elseif linsolve.cacheval isa Tuple && linsolve.cacheval[1] isa Factorization
            linsolve.cacheval[1]' \ dy
        elseif linsolve.alg isa AbstractKrylovSubspaceMethod
            # Doesn't modify `A`, so it's safe to just reuse it
            invprob = LinearSolve.LinearProblem(transpose(linsolve.A), dy)
            solve(invprob;
                abstol = linsolve.val.abstol,
                reltol = linsolve.val.reltol,
                verbose = linsolve.val.verbose,
                isfresh = freshbefore)
        else
            error("Algorithm $(linsolve.alg) is currently not supported by Enzyme rules on LinearSolve.jl. Please open an issue on LinearSolve.jl detailing which algorithm is missing the adjoint handling")
        end 

        dA .-= z * transpose(y)
        db .+= z
        dy .= eltype(dy)(0)
    end

    return (nothing,)
end

end