struct LinearCache{TA,Tb,Tu,Tp,Talg,Tc,Tl,Tr}
    A::TA
    b::Tb
    u::Tu
    p::Tp
    alg::Talg
    cacheval::Tc  # store alg cache here 
    isfresh::Bool # false => cacheval is set wrt A, true => update cacheval wrt A
    Pl::Tl        # store final preconditioner here. not being used rn
    Pr::Tr        # wrappers are using preconditioner in cache.alg for now
end

function set_A(cache::LinearCache, A)
    @set! cache.A = A
    @set! cache.isfresh = true
    return cache
end

function set_b(cache::LinearCache, b)
    @set! cache.b = b
    return cache
end

function set_u(cache::LinearCache, u)
    @set! cache.u = u
    return cache
end

function set_p(cache::LinearCache, p)
    @set! cache.p = p
#   @set! cache.isfresh = true
    return cache
end

function set_cacheval(cache::LinearCache, alg_cache)
    if cache.isfresh
        @set! cache.cacheval = alg_cache
        @set! cache.isfresh = false
    end
    return cache
end

init_cacheval(alg::SciMLLinearSolveAlgorithm, A, b, u) = nothing

function SciMLBase.init(prob::LinearProblem, alg, args...;
                        alias_A = false, alias_b = false,
                        kwargs...,
                       )
    @unpack A, b, u0, p = prob

    u0 = (u0 === nothing) ? zero(b) : u0

    cacheval = init_cacheval(alg, A, b, u0)
    isfresh = cacheval === nothing
    Tc = isfresh ? Any : typeof(cacheval)

    Pl = LinearAlgebra.I
    Pr = LinearAlgebra.I

    A = alias_A ? A : deepcopy(A)
    b = alias_b ? b : deepcopy(b)

    cache = LinearCache{
        typeof(A),
        typeof(b),
        typeof(u0),
        typeof(p),
        typeof(alg),
        Tc,
        typeof(Pl),
        typeof(Pr),
    }(
        A,
        b,
        u0,
        p,
        alg,
        cacheval,
        isfresh,
        Pl,
        Pr,
    )
    return cache
end

SciMLBase.solve(prob::LinearProblem, alg::SciMLLinearSolveAlgorithm,
                args...; kwargs...) = solve(init(prob, alg, args...; kwargs...))

SciMLBase.solve(cache::LinearCache, args...; kwargs...) =
    solve(cache, cache.alg, args...; kwargs...)

## make alg callable

function (alg::SciMLLinearSolveAlgorithm)(prob::LinearProblem,args...; kwargs...)
    x = solve(prob, alg, args...; kwargs...)
    return x
end

function (alg::SciMLLinearSolveAlgorithm)(x,A,b,args...;u0=nothing,kwargs...)
    prob = LinearProblem(A, b; u0=x)
    x = alg(prob, args...; kwargs...)
    return x
end

## make cache callable - and reuse

function (cache::LinearCache)(prob::LinearProblem, args...; kwargs...)

    if(prob.A != cache.A) cache = set_A(cache, prob.A) end
    if(prob.b != cache.b) cache = set_b(cache, prob.b) end

    if(prob.u0 == nothing)
        prob.u0 = zero(x)
    end

    cache = set_u(cache, prob.u0)
    x = solve(cache, args...; kwargs...)
    return x
end

function (cache::LinearCache)(x, A, b, args...; kwargs...)
    prob = LinearProblem(A, b; u0=x)
    x = cache(prob, args...; kwargs...)
    return x
end
