"""
$(SIGNATURES)
"""
function set_A(cache::LinearCache, A)
    @warn "set_A is deprecated for mutation on the cache. Use `cache.A = A"
    @set! cache.A = A
    @set! cache.isfresh = true
    return cache
end

"""
$(SIGNATURES)
"""
function set_b(cache::LinearCache, b)
    @warn "set_b is deprecated for mutation on the cache. Use `cache.b = b"
    @set! cache.b = b
    return cache
end

"""
$(SIGNATURES)
"""
function set_u(cache::LinearCache, u)
    @warn "set_u is deprecated for mutation on the cache. Use `cache.u = u"
    @set! cache.u = u
    return cache
end

"""
$(SIGNATURES)
"""
function set_p(cache::LinearCache, p)
    @warn "set_p is deprecated for mutation on the cache. Use `cache.p = p"
    @set! cache.p = p
    #   @set! cache.isfresh = true
    return cache
end

"""
$(SIGNATURES)
"""
function set_prec(cache, Pl, Pr)
    @warn "set_prec is deprecated for mutation on the cache. Use `cache.Pl = Pl; cache.Pr = Pr"
    @set! cache.Pl = Pl
    @set! cache.Pr = Pr
    return cache
end

function set_cacheval(cache::LinearCache, alg_cache)
    @warn "set_cacheval is deprecated for mutation on the cache. Use `cache.cacheval = cacheval; cache.isfresh = false"
    if cache.isfresh
        @set! cache.cacheval = alg_cache
        @set! cache.isfresh = false
        return cache
    end
end

@deprecate SciMLBase.solve(cache::LinearCache, args...; kwargs...) SciMLBase.solve!(cache::LinearCache, args...; kwargs...) false