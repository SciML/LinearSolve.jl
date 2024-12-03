module LinearSolveEnzymeExt

using LinearSolve
using LinearSolve.LinearAlgebra
using EnzymeCore
using EnzymeCore: EnzymeRules

@inline EnzymeCore.EnzymeRules.inactive_type(::Type{<:LinearSolve.SciMLLinearSolveAlgorithm}) = true

function EnzymeRules.forward(config::EnzymeRules.FwdConfigWidth{1},
        func::Const{typeof(LinearSolve.init)}, ::Type{RT}, prob::EnzymeCore.Annotation{LP},
        alg::Const; kwargs...) where {RT, LP <: LinearSolve.LinearProblem}
    @assert !(prob isa Const)
    res = func.val(prob.val, alg.val; kwargs...)
    if RT <: Const
        if EnzymeRules.needs_primal(config)
            return res
        else
            return nothing
        end
    end

    dres = func.val(prob.dval, alg.val; kwargs...)

    if dres.b == res.b
        dres.b .= false
    end
    if dres.A == res.A
        dres.A .= false
    end

    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        return Duplicated(res, dres)
    elseif EnzymeRules.needs_shadow(config)
        return dres
    elseif EnzymeRules.needs_primal(config)
        return res
    else
        return nothing
    end
end

function EnzymeRules.forward(
        config::EnzymeRules.FwdConfigWidth{1}, func::Const{typeof(LinearSolve.solve!)},
        ::Type{RT}, linsolve::EnzymeCore.Annotation{LP};
        kwargs...) where {RT, LP <: LinearSolve.LinearCache}
    @assert !(linsolve isa Const)

    res = func.val(linsolve.val; kwargs...)

    if RT <: Const
        if EnzymeRules.needs_primal(config)
            return res
        else
            return nothing
        end
    end
    if linsolve.val.alg isa LinearSolve.AbstractKrylovSubspaceMethod
        error("Algorithm $(_linsolve.alg) is currently not supported by Enzyme rules on LinearSolve.jl. Please open an issue on LinearSolve.jl detailing which algorithm is missing the adjoint handling")
    end

    res = deepcopy(res)  # Without this copy, the next solve will end up mutating the result

    b = linsolve.val.b
    linsolve.val.b = linsolve.dval.b - linsolve.dval.A * res.u
    dres = func.val(linsolve.val; kwargs...)
    linsolve.val.b = b

    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        return Duplicated(res, dres)
    elseif EnzymeRules.needs_shadow(config)
        return dres
    elseif EnzymeRules.needs_primal(config)
        return res
    else
        return nothing
    end
end

function EnzymeRules.augmented_primal(
        config, func::Const{typeof(LinearSolve.init)},
        ::Type{RT}, prob::EnzymeCore.Annotation{LP}, alg::Const;
        kwargs...) where {RT, LP <: LinearSolve.LinearProblem}
    res = func.val(prob.val, alg.val; kwargs...)
    dres = if EnzymeRules.width(config) == 1
        func.val(prob.dval, alg.val; kwargs...)
    else
        ntuple(Val(EnzymeRules.width(config))) do i
            Base.@_inline_meta
            func.val(prob.dval[i], alg.val; kwargs...)
        end
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

    prob_d_A = if EnzymeRules.width(config) == 1
        prob.dval.A
    else
        (dval.A for dval in prob.dval)
    end
    prob_d_b = if EnzymeRules.width(config) == 1
        prob.dval.b
    else
        (dval.b for dval in prob.dval)
    end

    return EnzymeRules.AugmentedReturn(res, dres, (d_A, d_b, prob_d_A, prob_d_b))
end

function EnzymeRules.reverse(
        config, func::Const{typeof(LinearSolve.init)}, ::Type{RT},
        cache, prob::EnzymeCore.Annotation{LP}, alg::Const;
        kwargs...) where {RT, LP <: LinearSolve.LinearProblem}
    d_A, d_b, prob_d_A, prob_d_b = cache

    if EnzymeRules.width(config) == 1
        if d_A !== prob_d_A
            prob_d_A .+= d_A
            d_A .= 0
        end
        if d_b !== prob_d_b
            prob_d_b .+= d_b
            d_b .= 0
        end
    else
        for (_prob_d_A, _d_A, _prob_d_b, _d_b) in zip(prob_d_A, d_A, prob_d_b, d_b)
            if _d_A !== _prob_d_A
                _prob_d_A .+= _d_A
                _d_A .= 0
            end
            if _d_b !== _prob_d_b
                _prob_d_b .+= _d_b
                _d_b .= 0
            end
        end
    end

    return (nothing, nothing)
end

# y=inv(A) B
#   dA −= z y^T  
#   dB += z, where  z = inv(A^T) dy
function EnzymeRules.augmented_primal(
        config, func::Const{typeof(LinearSolve.solve!)},
        ::Type{RT}, linsolve::EnzymeCore.Annotation{LP};
        kwargs...) where {RT, LP <: LinearSolve.LinearCache}
    res = func.val(linsolve.val; kwargs...)

    dres = if EnzymeRules.width(config) == 1
        deepcopy(res)
    else
        ntuple(Val(EnzymeRules.width(config))) do i
            Base.@_inline_meta
            deepcopy(res)
        end
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
        ntuple(Val(EnzymeRules.width(config))) do i
            Base.@_inline_meta
            dres[i].u
        end
    end

    dAs = if EnzymeRules.width(config) == 1
        (linsolve.dval.A,)
    else
        ntuple(Val(EnzymeRules.width(config))) do i
            Base.@_inline_meta
            linsolve.dval[i].A
        end
    end

    dbs = if EnzymeRules.width(config) == 1
        (linsolve.dval.b,)
    else
        ntuple(Val(EnzymeRules.width(config))) do i
            Base.@_inline_meta
            linsolve.dval[i].b
        end
    end

    cachesolve = deepcopy(linsolve.val)

    cache = (copy(res.u), resvals, cachesolve, dAs, dbs)
    return EnzymeRules.AugmentedReturn(res, dres, cache)
end

function EnzymeRules.reverse(config, func::Const{typeof(LinearSolve.solve!)},
        ::Type{RT}, cache, linsolve::EnzymeCore.Annotation{LP};
        kwargs...) where {RT, LP <: LinearSolve.LinearCache}
    y, dys, _linsolve, dAs, dbs = cache

    @assert !(linsolve isa Const)
    @assert !(linsolve isa Active)

    if EnzymeRules.width(config) == 1
        dys = (dys,)
    end

    for (dA, db, dy) in zip(dAs, dbs, dys)
        z = if _linsolve.cacheval isa Factorization
            _linsolve.cacheval' \ dy
        elseif _linsolve.cacheval isa Tuple && _linsolve.cacheval[1] isa Factorization
            _linsolve.cacheval[1]' \ dy
        elseif _linsolve.alg isa LinearSolve.AbstractKrylovSubspaceMethod
            # Doesn't modify `A`, so it's safe to just reuse it
            invprob = LinearSolve.LinearProblem(transpose(_linsolve.A), dy)
            solve(invprob, _linearsolve.alg;
                abstol = _linsolve.val.abstol,
                reltol = _linsolve.val.reltol,
                verbose = _linsolve.val.verbose)
        elseif _linsolve.alg isa LinearSolve.DefaultLinearSolver
            LinearSolve.defaultalg_adjoint_eval(_linsolve, dy)
        else
            error("Algorithm $(_linsolve.alg) is currently not supported by Enzyme rules on LinearSolve.jl. Please open an issue on LinearSolve.jl detailing which algorithm is missing the adjoint handling")
        end

        dA .-= z * transpose(y)
        db .+= z
        dy .= eltype(dy)(0)
    end

    return (nothing,)
end

end
