
struct LUFactorization{P} <: SciMLLinearSolveAlgorithm
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
    cache.A isa Union{AbstractMatrix,AbstractDiffEqOperator} ||
        error("LU is not defined for $(typeof(prob.A))")
    cache = set_cacheval(cache, lu!(cache.A, alg.pivot))
    ldiv!(cache.u,cache.cacheval, cache.b)
end

struct QRFactorization{P} <: SciMLLinearSolveAlgorithm
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
    cache.A isa Union{AbstractMatrix,AbstractDiffEqOperator} ||
        error("QR is not defined for $(typeof(prob.A))")
    cache = set_cacheval(
        cache,
        qr!(cache.A.A, alg.pivot; blocksize = alg.blocksize),
    )
    ldiv!(cache.u,cache.cacheval, cache.b)
end

struct SVDFactorization{A} <: SciMLLinearSolveAlgorithm
    full::Bool
    alg::A
end

SVDFactorization() = SVDFactorization(false, LinearAlgebra.DivideAndConquer())

function SciMLBase.solve(cache::LinearCache, alg::SVDFactorization)
    cache.A isa Union{AbstractMatrix,AbstractDiffEqOperator} ||
        error("SVD is not defined for $(typeof(prob.A))")
    cache = set_cacheval(cache, svd!(cache.A; full = alg.full, alg = alg.alg))
    ldiv!(cache.u,cache.cacheval, cache.b)
end
