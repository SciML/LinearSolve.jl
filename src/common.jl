struct LinearCache{TA,Tb,Tu,Tp,Talg,Tc,Tl,Tr}
    A::TA
    b::Tb
    u::Tu
    p::Tp
    alg::Talg
    cacheval::Tc # store alg cache here 
    isfresh::Bool
    Pl::Tl
    Pr::Tr
end

function set_A(cache, A) # and ! to function name
    @set! cache.A = A
    @set! cache.isfresh = true
    return cache
end

function set_b(cache, b)
    @set! cache.b = b
    return cache
end

function set_u(cache, u)
    @set! cache.u = u
    return cache
end

function set_p(cache, p)
    @set! cache.p = p
    # @set! cache.isfresh = true
    return cache
end

function set_cacheval(cache, alg_cache)
    if cache.isfresh
        @set! cache.cacheval = alg_cache
        @set! cache.isfresh = false
    end
    return cache
end

#function init_cacheval(cacheval, alg::SciMLLinearSolveAlgorithm)
#
#    return
#end

function SciMLBase.init(prob::LinearProblem, alg, args...;
                        alias_A = false, alias_b = false,
                        kwargs...,
                       )
    @unpack A, b, u0, p = prob

    if u0 == nothing
        u0 = zero(b)
    end

    if alg isa LUFactorization
        fact = lu_instance(A)
        Tfact = typeof(fact)
    else
        fact = nothing
        Tfact = Any
    end
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
        Tfact,
        typeof(Pl),
        typeof(Pr),
    }(
        A,
        b,
        u0,
        p,
        alg,
        fact,
        true,
        Pl,
        Pr,
    )
    return cache
end

SciMLBase.solve(prob::LinearProblem, alg, args...; kwargs...) =
    solve(init(prob, alg, args...; kwargs...))

SciMLBase.solve(cache) = solve(cache, cache.alg)

function (alg::SciMLLinearSolveAlgorithm)(prob::LinearProblem,args...;
                                          u0=nothing,kwargs...)
    x = solve(prob, alg, args...; kwargs...)
    return x
end

function (alg::SciMLLinearSolveAlgorithm)(x,A,b,args...;u0=nothing,kwargs...)
    prob = LinearProblem(A,b;u0=x)
    x = alg(prob, args...; kwargs...)
    return x
end

function (cache::LinearCache)(prob::LinearProblem,args...;u0=nothing,kwargs...)

    if prob.u0 == nothing
        prob.u0 = zero(x)
    end

    cache = set_A(cache, prob.A)
    cache = set_b(cache, prob.b)
    cache = set_u(cache, prob.u0)

    x = solve(cache,args...;kwargs...)
    return x
end

function (cache::LinearCache)(x,A,b,args...;u0=nothing,kwargs...)

    prob = LinearProblem(A,b;u0=x)
    x = cache(prob, args...; kwargs...)
    return x
end
