function SciMLBase.solve(cache::LinearCache, alg::AbstractFactorization; kwargs...)
    if cache.isfresh
        fact = init_cacheval(alg, cache.A, cache.b, cache.u)
        cache = set_cacheval(cache, fact)
    end

    y = ldiv!(cache.u, cache.cacheval, cache.b)
    SciMLBase.build_linear_solution(alg,y,nothing)
end

## LU Factorizations

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

    if A isa AbstractDiffEqOperator
        A = A.A
    end
    fact = lu!(A, alg.pivot)
    return fact
end

# This could be a GenericFactorization perhaps?
Base.@kwdef struct UMFPACKFactorization <: AbstractFactorization
    reuse_symbolic::Bool = true
end

function init_cacheval(::UMFPACKFactorization, A, b, u)
    if A isa AbstractDiffEqOperator
        A = A.A
    end
    if A isa SparseMatrixCSC
        return lu(A)
    else
        error("Sparse LU is not defined for $(typeof(A))")
    end
end

function SciMLBase.solve(cache::LinearCache, alg::UMFPACKFactorization)
    A = cache.A
    if A isa AbstractDiffEqOperator
        A = A.A
    end
    if cache.isfresh
        if cache.cacheval !== nothing && alg.reuse_symbolic
            # If we have a cacheval already, run umfpack_symbolic to ensure the symbolic factorization exists
            # This won't recompute if it does.
            SuiteSparse.UMFPACK.umfpack_symbolic!(cache.cacheval)
            fact = lu!(cache.cacheval, A)
        else
            fact = init_cacheval(alg, A, cache.b, cache.u)
        end
        cache = set_cacheval(cache, fact)
    end

    y = ldiv!(cache.u, cache.cacheval, cache.b)
    SciMLBase.build_linear_solution(alg,y,nothing)
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

    if A isa AbstractDiffEqOperator
        A = A.A
    end
    fact = qr!(A, alg.pivot; blocksize = alg.blocksize)
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

    if A isa AbstractDiffEqOperator
        A = A.A
    end

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

    if A isa AbstractDiffEqOperator
        A = A.A
    end
    fact = alg.fact_alg(A)
    return fact
end

## RFLUFactorization

RFLUFactorizaation() = GenericFactorization(;fact_alg=RecursiveFactorization.lu!)
