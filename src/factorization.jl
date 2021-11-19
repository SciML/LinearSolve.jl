
## LUFactorization

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

function init_cacheval(A, alg::LUFactorization)
    A isa Union{AbstractMatrix,AbstractDiffEqOperator} ||
        error("LU is not defined for $(typeof(A))")
    fact = lu!(A, alg.pivot)
    return fact
end

function SciMLBase.solve(cache::LinearCache, alg::LUFactorization)
    if cache.isfresh
        fact = init_cacheval(cache.A, alg)
        cache = set_cacheval(cache, fact)
    end

    ldiv!(cache.u,cache.cacheval, cache.b)
end

## QRFactorization

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

function init_cacheval(A, alg::QRFactorization)
    A isa Union{AbstractMatrix,AbstractDiffEqOperator} ||
        error("QR is not defined for $(typeof(A))")

    fact = qr!(A.A, alg.pivot; blocksize = alg.blocksize)
    return fact
end

function SciMLBase.solve(cache::LinearCache, alg::QRFactorization)
    if cache.isfresh
        fact = init_cacheval(cache.A, alg)
        cache = set_cacheval(cache, fact)
    end

    ldiv!(cache.u,cache.cacheval, cache.b)
end

## SVDFactorization

struct SVDFactorization{A} <: SciMLLinearSolveAlgorithm
    full::Bool
    alg::A
end

SVDFactorization() = SVDFactorization(false, LinearAlgebra.DivideAndConquer())

function init_cacheval(A, alg::SVDFactorization)
    A isa Union{AbstractMatrix,AbstractDiffEqOperator} ||
        error("SVD is not defined for $(typeof(A))")

    fact = svd!(A; full = alg.full, alg = alg.alg)
    return fact
end

function SciMLBase.solve(cache::LinearCache, alg::SVDFactorization)
    if cache.isfresh
        fact = init_cacheval(cache.A, alg)
        cache = set_cacheval(cache, fact)
    end

    ldiv!(cache.u,cache.cacheval, cache.b)
end
