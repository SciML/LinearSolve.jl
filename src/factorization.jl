function SciMLBase.solve(cache::LinearCache, alg::AbstractFactorization)
    if cache.isfresh
        fact = init_cacheval(alg, cache.A, cache.b, cache.u)
        cache = set_cacheval(cache, fact)
    end

    ldiv!(cache.u,cache.cacheval, cache.b)
end

## LUFactorization

struct LUFactorization{P} <: AbstractFactorization
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

function init_cacheval(alg::LUFactorization, A, b, u)
    A isa Union{AbstractMatrix,AbstractDiffEqOperator} ||
        error("LU is not defined for $(typeof(A))")
    fact = lu!(A, alg.pivot)
    return fact
end

## QRFactorization

struct QRFactorization{P} <: AbstractFactorization
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

function init_cacheval(alg::QRFactorization, A, b, u)
    A isa Union{AbstractMatrix,AbstractDiffEqOperator} ||
        error("QR is not defined for $(typeof(A))")

    fact = qr!(A.A, alg.pivot; blocksize = alg.blocksize)
    return fact
end

## SVDFactorization

struct SVDFactorization{A} <: AbstractFactorization
    full::Bool
    alg::A
end

SVDFactorization() = SVDFactorization(false, LinearAlgebra.DivideAndConquer())

function init_cacheval(alg::SVDFactorization, A, b, u)
    A isa Union{AbstractMatrix,AbstractDiffEqOperator} ||
        error("SVD is not defined for $(typeof(A))")

    fact = svd!(A; full = alg.full, alg = alg.alg)
    return fact
end

## GenericFactorization

struct GenericFactorization{F} <: AbstractFactorization
    fact_alg::F
end

GenericFactorization(;fact_alg = LinearAlgebra.factorize) =
    GenericFactorization(fact_alg)

function init_cacheval(alg::GenericFactorization, A, b, u)
    A isa Union{AbstractMatrix,AbstractDiffEqOperator} ||
        error("GenericFactorization is not defined for $(typeof(A))")

    fact = alg.fact_alg(A)
    return fact
end
