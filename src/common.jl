struct LinearCache{TA,Tb,Tu,Tp,Talg,Tc,Tl,Tr,Ttol}
    A::TA
    b::Tb
    u::Tu
    p::Tp
    alg::Talg
    cacheval::Tc  # store alg cache here
    isfresh::Bool # false => cacheval is set wrt A, true => update cacheval wrt A
    Pl::Tl        # store final preconditioner here. not being used rn
    Pr::Tr        # wrappers are using preconditioner in cache.alg for now
    abstol::Ttol
    reltol::Ttol
    maxiters::Int
    verbose::Bool
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

init_cacheval(alg::Union{SciMLLinearSolveAlgorithm,Nothing}, A, b, u) = nothing

SciMLBase.init(prob::LinearProblem, args...; kwargs...) = SciMLBase.init(prob,nothing,args...;kwargs...)

function SciMLBase.init(prob::LinearProblem, alg::Union{SciMLLinearSolveAlgorithm,Nothing}, args...;
                        alias_A = false, alias_b = false,
                        abstol=√eps(eltype(prob.A)),
                        reltol=√eps(eltype(prob.A)),
                        maxiters=length(prob.b),
                        verbose=false,
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
        typeof(reltol),
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
        abstol,
        reltol,
        maxiters,
        verbose,
    )
    return cache
end

SciMLBase.solve(prob::LinearProblem, args...; kwargs...) = solve(init(prob, nothing, args...; kwargs...))

SciMLBase.solve(prob::LinearProblem, alg::Union{SciMLLinearSolveAlgorithm,Nothing},
                args...; kwargs...) = solve(init(prob, alg, args...; kwargs...))

SciMLBase.solve(cache::LinearCache, args...; kwargs...) =
    solve(cache, cache.alg, args...; kwargs...)
