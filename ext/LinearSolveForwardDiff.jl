module LinearSolveForwardDiff

using LinearSolve
isdefined(Base, :get_extension) ? 
    (import ForwardDiff; using ForwardDiff: Dual) : 
    (import ..ForwardDiff; using ..ForwardDiff: Dual)

function LinearSolve.solve!(
            cache::LinearSolve.LinearCache{A_,B}, 
            alg::LinearSolve.AbstractFactorization; 
            kwargs...
        ) where {T, V, P, A_<:AbstractArray{<:Real}, B<:AbstractArray{<:Dual{T,V,P}}}
    @info "using solve! from LinearSolveForwardDiff.jl"
    dA = eltype(cache.A) <: Dual ? ForwardDiff.partials.(cache.A) : zero(cache.A)
    db = eltype(cache.b) <: Dual ? ForwardDiff.partials.(cache.b) : zero(cache.b)
    @show typeof(cache.A)
    @show typeof(cache.b)
    @show typeof(cache.u)
    A = eltype(cache.A) <: Dual ? ForwardDiff.value.(cache.A) : cache.A
    b = eltype(cache.b) <: Dual ? ForwardDiff.value.(cache.b) : cache.b
    u = eltype(cache.u) <: Dual ? ForwardDiff.value.(cache.u) : cache.u
    @show typeof(A), size(A)
    @show typeof(b), size(b)
    @show typeof(u), size(u)
    cache2 = remake(cache; A, b, u)
    res = LinearSolve.solve!(cache2, alg, kwargs...)
    dcache = remake(cache2; b = db - dA * res.u)
    dres = LinearSolve.solve!(dcache, alg, kwargs...)
    LinearSolve.SciMLBase.build_linear_solution(alg, Dual{T,V,P}.(res.u, dres.u), nothing, cache)
end

end # module 