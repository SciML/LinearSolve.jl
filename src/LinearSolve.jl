module LinearSolve

using Base: cache_dependencies, Bool
using SciMLBase: AbstractLinearAlgorithm, AbstractDiffEqOperator
using ArrayInterface: lu_instance
using UnPack
using Reexport
using LinearAlgebra
using Setfield
@reexport using SciMLBase

export LUFactorization, QRFactorization, SVDFactorization

#mutable?#
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

function set_cacheval(cache::LinearCache,alg)
    if cache.isfresh
        @set! cache.cacheval = alg
        @set! cache.isfresh = false
    end
    return cache
end

function SciMLBase.init(prob::LinearProblem, alg;
                        alias_A = false, alias_b = false,
                        kwargs...)
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

    cache = LinearCache{typeof(A),typeof(b),typeof(p),typeof(alg),Tfact,typeof(Pr),typeof(Pl)}(
        A, b, p, alg, fact, true, Pr, Pl
    )
    return cache
end

SciMLBase.solve(prob::LinearProblem, alg; kwargs...) = solve(init(prob, alg; kwargs...))
SciMLBase.solve(cache) = solve(cache, cache.alg)

struct LUFactorization{P} <: AbstractLinearAlgorithm
    pivot::P
end
function LUFactorization()
    pivot = @static if VERSION < v"1.7beta"
        Val(true)
    else
        RowMaximum()
    end
    LUFactorization(pivot)
end

function SciMLBase.solve(cache::LinearCache, alg::LUFactorization)
    cache.A isa Union{AbstractMatrix, AbstractDiffEqOperator} || error("LU is not defined for $(typeof(prob.A))")
    cache = set_cacheval(cache,lu!(cache.A, alg.pivot))
    ldiv!(cache.cacheval, cache.b)
end

struct QRFactorization{P} <: AbstractLinearAlgorithm
    pivot::P
    blocksize::Int
end
function QRFactorization()
    pivot = @static if VERSION < v"1.7beta"
        Val(false)
    else
        NoPivot()
    end
    QRFactorization(pivot, 16)
end

function SciMLBase.solve(cache::LinearCache, alg::QRFactorization)
    cache.A isa Union{AbstractMatrix, AbstractDiffEqOperator} || error("QR is not defined for $(typeof(prob.A))")
    cache = set_cacheval(cache,qr!(cache.A.A, alg.pivot; blocksize=alg.blocksize))
    ldiv!(cache.cacheval, cache.b)
end

struct SVDFactorization{A} <: AbstractLinearAlgorithm
    full::Bool
    alg::A
end
SVDFactorization() = SVDFactorization(false, LinearAlgebra.DivideAndConquer())

function SciMLBase.solve(cache::LinearCache, alg::SVDFactorization)
    cache.A isa Union{AbstractMatrix, AbstractDiffEqOperator} || error("SVD is not defined for $(typeof(prob.A))")
    cache = set_cacheval(cache,svd!(cache.A; full=alg.full, alg=alg.alg))
    ldiv!(cache.cacheval, cache.b)
end

end
