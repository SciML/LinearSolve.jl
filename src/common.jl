struct LinearCache{TA,Tb,Tu,Tp,Talg,Tc,Tr,Tl}
    A::TA
    b::Tb
    u::Tu
    p::Tp
    alg::Talg
    cacheval::Tc
    isfresh::Bool
    Pr::Tr
    Pl::Tl
#   k::Tk # iteration count
end

function set_A(cache, A) # and ! to function name
    @set! cache.A = A
    @set! cache.isfresh = true
end

function set_b(cache, b)
    @set! cache.b = b
end

function set_u(cache, u)
    @set! cache.u = u
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
    Pr = LinearAlgebra.I
    Pl = LinearAlgebra.I

    A = alias_A ? A : deepcopy(A)
    b = alias_b ? b : deepcopy(b)

    cache = LinearCache{
        typeof(A),
        typeof(b),
        typeof(u0),
        typeof(p),
        typeof(alg),
        Tfact,
        typeof(Pr),
        typeof(Pl),
    }(
        A,
        b,
        u0,
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

function (alg::SciMLLinearSolveAlgorithm)(x,A,b,args...;u0=nothing,kwargs...)
    prob = LinearProblem(A,b;u0=x)
    x = solve(prob,alg,args...;kwargs...)
    return x
end

# how to initialize cahce?

# use the same cache to solve multiple linear problems
function (cache::LinearCache)(x,A,b,args...;u0=nothing,kwargs...)
    set_A(cache, A)
    set_b(cache, b)

    if u0 == nothing
        x = zero(x)
    else
        x = u0
    end
    set_u(cache, x)

    x = solve(cache)
    return x
end
