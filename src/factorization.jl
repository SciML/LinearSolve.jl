
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
    fact = lu!(cache.A, alg.pivot)
    cache = set_cacheval(cache, fact)
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
    fact = qr!(cache.A.A, alg.pivot; blocksize = alg.blocksize)
    cache = set_cacheval(cache, fact)
    ldiv!(cache.u,cache.cacheval, cache.b)
end

struct SVDFactorization{A} <: SciMLLinearSolveAlgorithm
    full::Bool
    alg::A
end

SVDFactorization() = SVDFactorization(false, LinearAlgebra.DivideAndConquer())

function SciMLBase.solve(cache::LinearCache, alg::SVDFactorization)
    cache.A isa Union{AbstractMatrix,AbstractDiffEqOperator} ||
        error("SVD is not defined for $(typeof(cache.A))")
    fact = svd!(cache.A; full = alg.full, alg = alg.alg)
    cache = set_cacheval(cache, fact)
    ldiv!(cache.u,cache.cacheval, cache.b)
end
