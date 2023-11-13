module LinearSolveForwardDiff

using LinearSolve
isdefined(Base, :get_extension) ? 
    (import ForwardDiff; using ForwardDiff: Dual) : 
    (import ..ForwardDiff; using ..ForwardDiff: Dual)

function _solve!(cache, alg, dAs, dbs, A, b, T; kwargs...)
    @assert !(eltype(first(dAs)) isa Dual)
    @assert !(eltype(first(dbs)) isa Dual)
    @assert !(eltype(A) isa Dual)
    @assert !(eltype(b) isa Dual)
    reltol = cache.reltol isa Dual ? ForwardDiff.value(cache.reltol) : cache.reltol
    abstol = cache.abstol isa Dual ? ForwardDiff.value(cache.abstol) : cache.abstol
    u = eltype(cache.u) <: Dual ? ForwardDiff.value.(cache.u) : cache.u
    cacheval = eltype(cache.cacheval.factors) <: Dual ? begin 
        LinearSolve.LinearAlgebra.LU(ForwardDiff.value.(cache.cacheval.factors), cache.cacheval.ipiv, cache.cacheval.info)
    end : cache.cacheval
    cache2 = remake(cache; A, b, u, reltol, abstol, cacheval)
    res = LinearSolve.solve!(cache2, alg, kwargs...)
    dresus = reduce(hcat, map(dAs, dbs) do dA, db
        cache2.b = db - dA * res.u
        dres = LinearSolve.solve!(cache2, alg, kwargs...)
        deepcopy(dres.u)
    end)
    # display(dresus)
    d = Dual{T}.(res.u, Tuple.(eachrow(dresus)))
    LinearSolve.SciMLBase.build_linear_solution(alg, d, nothing, cache; retcode=res.retcode, iters=res.iters, stats=res.stats)
end

function LinearSolve.solve!(
            cache::LinearSolve.LinearCache{<:AbstractMatrix{<:Dual{T,V,P}}}, 
            alg::LinearSolve.AbstractFactorization; 
            kwargs...
        ) where {T, V, P}
    @info "using solve! df/dA"
    dAs = begin
        dAs_ = ForwardDiff.partials.(cache.A)
        dAs_ = collect.(dAs_)
        dAs_ = [getindex.(dAs_, i) for i in 1:length(first(dAs_))]
    end
    dbs = [zero(cache.b) for _=1:P]
    A = ForwardDiff.value.(cache.A)
    b = cache.b
    _solve!(cache, alg, dAs, dbs, A, b, T; kwargs...)
end
function LinearSolve.solve!(
            cache::LinearSolve.LinearCache{A_,<:AbstractArray{<:Dual{T,V,P}}}, 
            alg::LinearSolve.AbstractFactorization; 
            kwargs...
        ) where {T, V, P, A_}
    @info "using solve! df/db"
    dAs = [zero(cache.A) for _=1:P]
    dbs = begin
        dbs_ = ForwardDiff.partials.(cache.b)
        dbs_ = collect.(dbs_)
        dbs_ = [getindex.(dbs_, i) for i in 1:length(first(dbs_))]
    end
    A = cache.A
    b = ForwardDiff.value.(cache.b)
    _solve!(cache, alg, dAs, dbs, A, b, T; kwargs...)
end

end # module 