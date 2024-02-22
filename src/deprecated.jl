const warned_a = Ref(false)
const warned_b = Ref(false)
const warned_u = Ref(false)
const warned_p = Ref(false)
const warned_prec = Ref(false)
const warned_cacheval = Ref(false)
"""
$(SIGNATURES)
"""
function set_A(cache::LinearCache, A)
    if !warned_a[]
        @warn "set_A is deprecated for mutation on the cache. Use `cache.A = A"
        warned_a[] = true
    end
    @set! cache.A = A
    @set! cache.isfresh = true
    return cache
end

"""
$(SIGNATURES)
"""
function set_b(cache::LinearCache, b)
    if !warned_b[]
        @warn "set_b is deprecated for mutation on the cache. Use `cache.b = b"
        warned_b[] = true
    end
    @set! cache.b = b
    return cache
end

"""
$(SIGNATURES)
"""
function set_u(cache::LinearCache, u)
    if !warned_u[]
        @warn "set_u is deprecated for mutation on the cache. Use `cache.u = u"
        warned_u[] = true
    end
    @set! cache.u = u
    return cache
end

"""
$(SIGNATURES)
"""
function set_p(cache::LinearCache, p)
    if !warned_p[]
        @warn "set_p is deprecated for mutation on the cache. Use `cache.p = p"
        warned_p[] = true
    end
    @set! cache.p = p
    #   @set! cache.isfresh = true
    return cache
end

"""
$(SIGNATURES)
"""
function set_prec(cache, Pl, Pr)
    if !warned_prec[]
        @warn "set_prec is deprecated for mutation on the cache. Use `cache.Pl = Pl; cache.Pr = Pr"
        warned_prec[] = true
    end
    @set! cache.Pl = Pl
    @set! cache.Pr = Pr
    return cache
end

function set_cacheval(cache::LinearCache, alg_cache)
    if !warned_cacheval[]
        @warn "set_cacheval is deprecated for mutation on the cache. Use `cache.cacheval = cacheval; cache.isfresh = false"
        warned_cacheval[] = true
    end
    if cache.isfresh
        @set! cache.cacheval = alg_cache
        @set! cache.isfresh = false
        return cache
    end
end

@deprecate SciMLBase.solve(cache::LinearCache, args...; kwargs...) SciMLBase.solve!(
    cache::LinearCache,
    args...;
    kwargs...) false
