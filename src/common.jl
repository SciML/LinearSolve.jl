struct LinearCache{TA,Tb,Tp,Talg,Tc,Tr,Tl}
    A::TA
    b::Tb
    p::Tp
    alg::Talg
    cacheval::Tc
    isfresh::Bool
    Pr::Tr
    Pl::Tl
end

function set_A(cache, A)
    @set! cache.A = A
    @set! cache.isfresh = true
end

function set_b(cache, b)
    @set! cache.b = b
end

function set_p(cache, p)
    @set! cache.p = p
    # @set! cache.isfresh = true
end

function set_cacheval(cache::LinearCache, alg)
    if cache.isfresh
        @set! cache.cacheval = alg
        @set! cache.isfresh = false
    end
    return cache
end

function SciMLBase.init(
    prob::LinearProblem,
    alg,
    args...;
    alias_A = false,
    alias_b = false,
    kwargs...,
)
    @unpack A, b, p = prob
    if alg isa LUFactorization
        fact = lu_instance(A)
        Tfact = typeof(fact)
    else
        fact = nothing
        Tfact = Any
    end
    Pr = nothing
    Pl = nothing

    A = alias_A ? A : copy(A)
    b = alias_b ? b : copy(b)

    cache = LinearCache{
        typeof(A),
        typeof(b),
        typeof(p),
        typeof(alg),
        Tfact,
        typeof(Pr),
        typeof(Pl),
    }(
        A,
        b,
        p,
        alg,
        fact,
        true,
        Pr,
        Pl,
    )
    return cache
end

SciMLBase.solve(prob::LinearProblem, alg, args...; kwargs...) =
    solve(init(prob, alg, args...; kwargs...))

SciMLBase.solve(cache) = solve(cache, cache.alg)
