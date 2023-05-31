_ldiv!(x, A, b) = ldiv!(x, A, b)

function _ldiv!(x::Vector, A::Factorization, b::Vector)
    # workaround https://github.com/JuliaLang/julia/issues/43507
    copyto!(x, b)
    ldiv!(A, x)
end

@generated function SciMLBase.solve!(cache::LinearCache, alg::AbstractFactorization;
                                     kwargs...)
    quote
        if cache.isfresh
            fact = do_factorization(alg, cache.A, cache.b, cache.u)
            cache.cacheval = fact
            cache.isfresh = false
        end
        y = _ldiv!(cache.u, get_cacheval(cache, $(Meta.quot(defaultalg_symbol(alg)))),
                   cache.b)

        #=
        retcode = if LinearAlgebra.issuccess(fact)
            SciMLBase.ReturnCode.Success
        else
            SciMLBase.ReturnCode.Failure
        end
        SciMLBase.build_linear_solution(alg, y, nothing, cache; retcode = retcode)
        =#
        SciMLBase.build_linear_solution(alg, y, nothing, cache)
    end
end

#RF Bad fallback: will fail if `A` is just a stand-in
# This should instead just create the factorization type.
function init_cacheval(alg::AbstractFactorization, A, b, u, Pl, Pr, maxiters::Int, abstol,
                       reltol, verbose::Bool, assumptions::OperatorAssumptions)
    do_factorization(alg, convert(AbstractMatrix, A), b, u)
end

## LU Factorizations

struct LUFactorization{P} <: AbstractFactorization
    pivot::P
end

struct GenericLUFactorization{P} <: AbstractFactorization
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

function GenericLUFactorization()
    pivot = @static if VERSION < v"1.7beta"
        Val(true)
    else
        RowMaximum()
    end
    GenericLUFactorization(pivot)
end

function do_factorization(alg::LUFactorization, A, b, u)
    A = convert(AbstractMatrix, A)
    if A isa AbstractSparseMatrixCSC
        return lu(SparseMatrixCSC(size(A)..., getcolptr(A), rowvals(A), nonzeros(A)))
    else
        fact = lu!(A, alg.pivot)
    end
    return fact
end

function do_factorization(alg::GenericLUFactorization, A, b, u)
    A = convert(AbstractMatrix, A)
    fact = LinearAlgebra.generic_lufact!(A, alg.pivot)
    return fact
end

function init_cacheval(alg::Union{LUFactorization, GenericLUFactorization}, A, b, u, Pl, Pr,
                       maxiters::Int, abstol, reltol, verbose::Bool,
                       assumptions::OperatorAssumptions)
    ArrayInterface.lu_instance(convert(AbstractMatrix, A))
end

const PREALLOCATED_LU = ArrayInterface.lu_instance(rand(1, 1))

function init_cacheval(alg::Union{LUFactorization, GenericLUFactorization},
                       A::Matrix{Float64}, b, u, Pl, Pr,
                       maxiters::Int, abstol, reltol, verbose::Bool,
                       assumptions::OperatorAssumptions)
    PREALLOCATED_LU
end

function init_cacheval(alg::Union{LUFactorization, GenericLUFactorization},
                       A::AbstractSciMLOperator, b, u, Pl, Pr,
                       maxiters::Int, abstol, reltol, verbose::Bool,
                       assumptions::OperatorAssumptions)
    nothing
end

@static if VERSION < v"1.7-"
    function init_cacheval(alg::Union{LUFactorization, GenericLUFactorization},
                           A::Union{Diagonal, SymTridiagonal}, b, u, Pl, Pr,
                           maxiters::Int, abstol, reltol, verbose::Bool,
                           assumptions::OperatorAssumptions)
        nothing
    end
end

## QRFactorization

struct QRFactorization{P} <: AbstractFactorization
    pivot::P
    blocksize::Int
    inplace::Bool
end

function QRFactorization(inplace = true)
    pivot = @static if VERSION < v"1.7beta"
        Val(false)
    else
        NoPivot()
    end
    QRFactorization(pivot, 16, inplace)
end

function do_factorization(alg::QRFactorization, A, b, u)
    A = convert(AbstractMatrix, A)
    if alg.inplace && !(A isa SparseMatrixCSC) && !(A isa GPUArraysCore.AbstractGPUArray)
        fact = qr!(A, alg.pivot)
    else
        fact = qr(A) # CUDA.jl does not allow other args!
    end
    return fact
end

function init_cacheval(alg::QRFactorization, A, b, u, Pl, Pr,
                       maxiters::Int, abstol, reltol, verbose::Bool,
                       assumptions::OperatorAssumptions)
    ArrayInterface.qr_instance(convert(AbstractMatrix, A))
end

const PREALLOCATED_QR = ArrayInterface.qr_instance(rand(1, 1))

function init_cacheval(alg::QRFactorization, A::Matrix{Float64}, b, u, Pl, Pr,
                       maxiters::Int, abstol, reltol, verbose::Bool,
                       assumptions::OperatorAssumptions)
    PREALLOCATED_QR
end

function init_cacheval(alg::QRFactorization, A::AbstractSciMLOperator, b, u, Pl, Pr,
                       maxiters::Int, abstol, reltol, verbose::Bool,
                       assumptions::OperatorAssumptions)
    nothing
end

@static if VERSION < v"1.7-"
    function init_cacheval(alg::QRFactorization,
                           A::Union{Diagonal, SymTridiagonal, Tridiagonal}, b, u, Pl, Pr,
                           maxiters::Int, abstol, reltol, verbose::Bool,
                           assumptions::OperatorAssumptions)
        nothing
    end
end

## CholeskyFactorization

struct CholeskyFactorization{P, P2} <: AbstractFactorization
    pivot::P
    tol::Int
    shift::Float64
    perm::P2
end

function CholeskyFactorization(; pivot = nothing, tol = 0.0, shift = 0.0, perm = nothing)
    if pivot === nothing
        pivot = @static if VERSION < v"1.7beta"
            Val(false)
        else
            NoPivot()
        end
    end
    CholeskyFactorization(pivot, 16, shift, perm)
end

function do_factorization(alg::CholeskyFactorization, A, b, u)
    A = convert(AbstractMatrix, A)
    if A isa SparseMatrixCSC
        fact = cholesky!(A; shift = alg.shift, check = false, perm = alg.perm)
    elseif alg.pivot === Val(false) || alg.pivot === NoPivot()
        fact = cholesky!(A, alg.pivot; check = false)
    else
        fact = cholesky!(A, alg.pivot; tol = alg.tol, check = false)
    end
    return fact
end

function init_cacheval(alg::CholeskyFactorization, A, b, u, Pl, Pr,
                       maxiters::Int, abstol, reltol, verbose::Bool,
                       assumptions::OperatorAssumptions)
    ArrayInterface.cholesky_instance(convert(AbstractMatrix, A), alg.pivot)
end

@static if VERSION < v"1.7beta"
    cholpivot = Val(false)
else
    cholpivot = NoPivot()
end

const PREALLOCATED_CHOLESKY = ArrayInterface.cholesky_instance(rand(1, 1), cholpivot)

function init_cacheval(alg::CholeskyFactorization, A::Matrix{Float64}, b, u, Pl, Pr,
                       maxiters::Int, abstol, reltol, verbose::Bool,
                       assumptions::OperatorAssumptions)
    PREALLOCATED_CHOLESKY
end

function init_cacheval(alg::CholeskyFactorization, A::Diagonal, b, u, Pl, Pr,
                       maxiters::Int, abstol, reltol, verbose::Bool,
                       assumptions::OperatorAssumptions)
    nothing
end

function init_cacheval(alg::CholeskyFactorization, A::AbstractSciMLOperator, b, u, Pl, Pr,
                       maxiters::Int, abstol, reltol, verbose::Bool,
                       assumptions::OperatorAssumptions)
    nothing
end

@static if VERSION < v"1.7beta"
    function init_cacheval(alg::CholeskyFactorization,
                           A::Union{SymTridiagonal, Tridiagonal}, b, u, Pl, Pr,
                           maxiters::Int, abstol, reltol, verbose::Bool,
                           assumptions::OperatorAssumptions)
        nothing
    end
end

## LDLtFactorization

struct LDLtFactorization{T} <: AbstractFactorization
    shift::Float64
    perm::T
end

function LDLtFactorization(shift = 0.0, perm = nothing)
    LDLtFactorization(shift, perm)
end

function do_factorization(alg::LDLtFactorization, A, b, u)
    A = convert(AbstractMatrix, A)
    if !(A isa SparseMatrixCSC)
        fact = ldlt!(A)
    else
        fact = ldlt!(A, shift = alg.shift, perm = alg.perm)
    end
    return fact
end

function init_cacheval(alg::LDLtFactorization, A, b, u, Pl, Pr,
                       maxiters::Int, abstol, reltol,
                       verbose::Bool, assumptions::OperatorAssumptions)
    nothing
end

function init_cacheval(alg::LDLtFactorization, A::SymTridiagonal, b, u, Pl, Pr,
                       maxiters::Int, abstol, reltol, verbose::Bool,
                       assumptions::OperatorAssumptions)
    ArrayInterface.ldlt_instance(convert(AbstractMatrix, A))
end

## SVDFactorization

struct SVDFactorization{A} <: AbstractFactorization
    full::Bool
    alg::A
end

SVDFactorization() = SVDFactorization(false, LinearAlgebra.DivideAndConquer())

function do_factorization(alg::SVDFactorization, A, b, u)
    A = convert(AbstractMatrix, A)
    fact = svd!(A; full = alg.full, alg = alg.alg)
    return fact
end

function init_cacheval(alg::SVDFactorization, A::Matrix, b, u, Pl, Pr,
                       maxiters::Int, abstol, reltol, verbose::Bool,
                       assumptions::OperatorAssumptions)
    ArrayInterface.svd_instance(convert(AbstractMatrix, A))
end

const PREALLOCATED_SVD = ArrayInterface.svd_instance(rand(1, 1))

function init_cacheval(alg::SVDFactorization, A::Matrix{Float64}, b, u, Pl, Pr,
                       maxiters::Int, abstol, reltol, verbose::Bool,
                       assumptions::OperatorAssumptions)
    PREALLOCATED_SVD
end

function init_cacheval(alg::SVDFactorization, A, b, u, Pl, Pr,
                       maxiters::Int, abstol, reltol, verbose::Bool,
                       assumptions::OperatorAssumptions)
    nothing
end

@static if VERSION < v"1.7-"
    function init_cacheval(alg::SVDFactorization,
                           A::Union{Diagonal, SymTridiagonal, Tridiagonal}, b, u, Pl, Pr,
                           maxiters::Int, abstol, reltol, verbose::Bool,
                           assumptions::OperatorAssumptions)
        nothing
    end
end

## BunchKaufmanFactorization

Base.@kwdef struct BunchKaufmanFactorization <: AbstractFactorization
    rook::Bool = false
end

function do_factorization(alg::BunchKaufmanFactorization, A, b, u)
    A = convert(AbstractMatrix, A)
    fact = bunchkaufman!(A, alg.rook; check = false)
    return fact
end

function init_cacheval(alg::BunchKaufmanFactorization, A::Symmetric{<:Number, <:Matrix}, b,
                       u, Pl, Pr,
                       maxiters::Int, abstol, reltol, verbose::Bool,
                       assumptions::OperatorAssumptions)
    ArrayInterface.bunchkaufman_instance(convert(AbstractMatrix, A))
end

const PREALLOCATED_BUNCHKAUFMAN = ArrayInterface.bunchkaufman_instance(Symmetric(rand(1, 1)))

function init_cacheval(alg::BunchKaufmanFactorization,
                       A::Symmetric{Float64, Matrix{Float64}}, b, u, Pl, Pr,
                       maxiters::Int, abstol, reltol, verbose::Bool,
                       assumptions::OperatorAssumptions)
    PREALLOCATED_BUNCHKAUFMAN
end

function init_cacheval(alg::BunchKaufmanFactorization, A, b, u, Pl, Pr,
                       maxiters::Int, abstol, reltol, verbose::Bool,
                       assumptions::OperatorAssumptions)
    nothing
end

## GenericFactorization

struct GenericFactorization{F} <: AbstractFactorization
    fact_alg::F
end

GenericFactorization(; fact_alg = LinearAlgebra.factorize) = GenericFactorization(fact_alg)

function do_factorization(alg::GenericFactorization, A, b, u)
    A = convert(AbstractMatrix, A)
    fact = alg.fact_alg(A)
    return fact
end

function init_cacheval(alg::GenericFactorization{typeof(lu)}, A, b, u, Pl, Pr,
                       maxiters::Int,
                       abstol, reltol, verbose::Bool, assumptions::OperatorAssumptions)
    ArrayInterface.lu_instance(convert(AbstractMatrix, A))
end
function init_cacheval(alg::GenericFactorization{typeof(lu!)}, A, b, u, Pl, Pr,
                       maxiters::Int,
                       abstol, reltol, verbose::Bool, assumptions::OperatorAssumptions)
    ArrayInterface.lu_instance(convert(AbstractMatrix, A))
end

function init_cacheval(alg::GenericFactorization{typeof(lu)},
                       A::StridedMatrix{<:LinearAlgebra.BlasFloat}, b, u, Pl, Pr,
                       maxiters::Int,
                       abstol, reltol, verbose::Bool, assumptions::OperatorAssumptions)
    ArrayInterface.lu_instance(A)
end
function init_cacheval(alg::GenericFactorization{typeof(lu!)},
                       A::StridedMatrix{<:LinearAlgebra.BlasFloat}, b, u, Pl, Pr,
                       maxiters::Int,
                       abstol, reltol, verbose::Bool, assumptions::OperatorAssumptions)
    ArrayInterface.lu_instance(A)
end
function init_cacheval(alg::GenericFactorization{typeof(lu)}, A::Diagonal, b, u, Pl, Pr,
                       maxiters::Int, abstol, reltol, verbose::Bool,
                       assumptions::OperatorAssumptions)
    Diagonal(inv.(A.diag))
end
function init_cacheval(alg::GenericFactorization{typeof(lu)}, A::Tridiagonal, b, u, Pl, Pr,
                       maxiters::Int, abstol, reltol, verbose::Bool,
                       assumptions::OperatorAssumptions)
    ArrayInterface.lu_instance(A)
end
function init_cacheval(alg::GenericFactorization{typeof(lu!)}, A::Diagonal, b, u, Pl, Pr,
                       maxiters::Int, abstol, reltol, verbose::Bool,
                       assumptions::OperatorAssumptions)
    Diagonal(inv.(A.diag))
end
function init_cacheval(alg::GenericFactorization{typeof(lu!)}, A::Tridiagonal, b, u, Pl, Pr,
                       maxiters::Int, abstol, reltol, verbose::Bool,
                       assumptions::OperatorAssumptions)
    ArrayInterface.lu_instance(A)
end

function init_cacheval(alg::GenericFactorization{typeof(qr)}, A, b, u, Pl, Pr,
                       maxiters::Int,
                       abstol, reltol, verbose::Bool, assumptions::OperatorAssumptions)
    ArrayInterface.qr_instance(convert(AbstractMatrix, A))
end
function init_cacheval(alg::GenericFactorization{typeof(qr!)}, A, b, u, Pl, Pr,
                       maxiters::Int,
                       abstol, reltol, verbose::Bool, assumptions::OperatorAssumptions)
    ArrayInterface.qr_instance(convert(AbstractMatrix, A))
end

function init_cacheval(alg::GenericFactorization{typeof(qr)},
                       A::StridedMatrix{<:LinearAlgebra.BlasFloat}, b, u, Pl, Pr,
                       maxiters::Int,
                       abstol, reltol, verbose::Bool, assumptions::OperatorAssumptions)
    ArrayInterface.qr_instance(A)
end
function init_cacheval(alg::GenericFactorization{typeof(qr!)},
                       A::StridedMatrix{<:LinearAlgebra.BlasFloat}, b, u, Pl, Pr,
                       maxiters::Int,
                       abstol, reltol, verbose::Bool, assumptions::OperatorAssumptions)
    ArrayInterface.qr_instance(A)
end
function init_cacheval(alg::GenericFactorization{typeof(qr)}, A::Diagonal, b, u, Pl, Pr,
                       maxiters::Int, abstol, reltol, verbose::Bool,
                       assumptions::OperatorAssumptions)
    Diagonal(inv.(A.diag))
end
function init_cacheval(alg::GenericFactorization{typeof(qr)}, A::Tridiagonal, b, u, Pl, Pr,
                       maxiters::Int, abstol, reltol, verbose::Bool,
                       assumptions::OperatorAssumptions)
    ArrayInterface.qr_instance(A)
end
function init_cacheval(alg::GenericFactorization{typeof(qr!)}, A::Diagonal, b, u, Pl, Pr,
                       maxiters::Int, abstol, reltol, verbose::Bool,
                       assumptions::OperatorAssumptions)
    Diagonal(inv.(A.diag))
end
function init_cacheval(alg::GenericFactorization{typeof(qr!)}, A::Tridiagonal, b, u, Pl, Pr,
                       maxiters::Int, abstol, reltol, verbose::Bool,
                       assumptions::OperatorAssumptions)
    ArrayInterface.qr_instance(A)
end

function init_cacheval(alg::GenericFactorization{typeof(svd)}, A, b, u, Pl, Pr,
                       maxiters::Int,
                       abstol, reltol, verbose::Bool, assumptions::OperatorAssumptions)
    ArrayInterface.svd_instance(convert(AbstractMatrix, A))
end
function init_cacheval(alg::GenericFactorization{typeof(svd!)}, A, b, u, Pl, Pr,
                       maxiters::Int,
                       abstol, reltol, verbose::Bool, assumptions::OperatorAssumptions)
    ArrayInterface.svd_instance(convert(AbstractMatrix, A))
end

function init_cacheval(alg::GenericFactorization{typeof(svd)},
                       A::StridedMatrix{<:LinearAlgebra.BlasFloat}, b, u, Pl, Pr,
                       maxiters::Int,
                       abstol, reltol, verbose::Bool, assumptions::OperatorAssumptions)
    ArrayInterface.svd_instance(A)
end
function init_cacheval(alg::GenericFactorization{typeof(svd!)},
                       A::StridedMatrix{<:LinearAlgebra.BlasFloat}, b, u, Pl, Pr,
                       maxiters::Int,
                       abstol, reltol, verbose::Bool, assumptions::OperatorAssumptions)
    ArrayInterface.svd_instance(A)
end
function init_cacheval(alg::GenericFactorization{typeof(svd)}, A::Diagonal, b, u, Pl, Pr,
                       maxiters::Int, abstol, reltol, verbose::Bool,
                       assumptions::OperatorAssumptions)
    Diagonal(inv.(A.diag))
end
function init_cacheval(alg::GenericFactorization{typeof(svd)}, A::Tridiagonal, b, u, Pl, Pr,
                       maxiters::Int, abstol, reltol, verbose::Bool,
                       assumptions::OperatorAssumptions)
    ArrayInterface.svd_instance(A)
end
function init_cacheval(alg::GenericFactorization{typeof(svd!)}, A::Diagonal, b, u, Pl, Pr,
                       maxiters::Int, abstol, reltol, verbose::Bool,
                       assumptions::OperatorAssumptions)
    Diagonal(inv.(A.diag))
end
function init_cacheval(alg::GenericFactorization{typeof(svd!)}, A::Tridiagonal, b, u, Pl,
                       Pr,
                       maxiters::Int, abstol, reltol, verbose::Bool,
                       assumptions::OperatorAssumptions)
    ArrayInterface.svd_instance(A)
end

function init_cacheval(alg::GenericFactorization, A::Diagonal, b, u, Pl, Pr, maxiters::Int,
                       abstol, reltol, verbose::Bool, assumptions::OperatorAssumptions)
    Diagonal(inv.(A.diag))
end
function init_cacheval(alg::GenericFactorization, A::Tridiagonal, b, u, Pl, Pr,
                       maxiters::Int,
                       abstol, reltol, verbose::Bool, assumptions::OperatorAssumptions)
    ArrayInterface.lu_instance(A)
end
function init_cacheval(alg::GenericFactorization, A::SymTridiagonal{T, V}, b, u, Pl, Pr,
                       maxiters::Int, abstol, reltol, verbose::Bool,
                       assumptions::OperatorAssumptions) where {T, V}
    LinearAlgebra.LDLt{T, SymTridiagonal{T, V}}(A)
end

function init_cacheval(alg::Union{GenericFactorization{typeof(bunchkaufman!)},
                                  GenericFactorization{typeof(bunchkaufman)}},
                       A::Union{Hermitian, Symmetric}, b, u, Pl, Pr, maxiters::Int, abstol,
                       reltol, verbose::Bool, assumptions::OperatorAssumptions)
    BunchKaufman(A.data, Array(1:size(A, 1)), A.uplo, true, false, 0)
end

function init_cacheval(alg::Union{GenericFactorization{typeof(bunchkaufman!)},
                                  GenericFactorization{typeof(bunchkaufman)}},
                       A::StridedMatrix{<:LinearAlgebra.BlasFloat}, b, u, Pl, Pr,
                       maxiters::Int,
                       abstol, reltol, verbose::Bool, assumptions::OperatorAssumptions)
    if eltype(A) <: Complex
        return bunchkaufman!(Hermitian(A))
    else
        return bunchkaufman!(Symmetric(A))
    end
end

# Fallback, tries to make nonsingular and just factorizes
# Try to never use it.

# Cholesky needs the posdef matrix, for GenericFactorization assume structure is needed
function init_cacheval(alg::Union{GenericFactorization{typeof(cholesky)},
                                  GenericFactorization{typeof(cholesky!)}}, A, b, u, Pl, Pr,
                       maxiters::Int, abstol, reltol, verbose::Bool,
                       assumptions::OperatorAssumptions)
    newA = copy(convert(AbstractMatrix, A))
    do_factorization(alg, newA, b, u)
end

function init_cacheval(alg::Union{GenericFactorization},
                       A::Union{Hermitian{T, <:SparseMatrixCSC},
                                Symmetric{T, <:SparseMatrixCSC}}, b, u, Pl, Pr,
                       maxiters::Int, abstol, reltol, verbose::Bool,
                       assumptions::OperatorAssumptions) where {T}
    newA = copy(convert(AbstractMatrix, A))
    do_factorization(alg, newA, b, u)
end

# Ambiguity handling dispatch

################################## Factorizations which require solve! overloads

Base.@kwdef struct UMFPACKFactorization <: AbstractFactorization
    reuse_symbolic::Bool = true
    check_pattern::Bool = true # Check factorization re-use
end

@static if VERSION < v"1.9.0-DEV.1622"
    const PREALLOCATED_UMFPACK = SuiteSparse.UMFPACK.UmfpackLU(C_NULL, C_NULL, 0, 0,
                                                               [0], Int64[], Float64[], 0)
    finalizer(SuiteSparse.UMFPACK.umfpack_free_symbolic, PREALLOCATED_UMFPACK)
else
    const PREALLOCATED_UMFPACK = SuiteSparse.UMFPACK.UmfpackLU(SparseMatrixCSC(0, 0, [1],
                                                                               Int64[],
                                                                               Float64[]))
end

function init_cacheval(alg::UMFPACKFactorization,
                       A::Union{Nothing, Matrix, AbstractSciMLOperator}, b, u, Pl, Pr,
                       maxiters::Int, abstol, reltol,
                       verbose::Bool, assumptions::OperatorAssumptions)
    nothing
end

function init_cacheval(alg::UMFPACKFactorization, A::SparseMatrixCSC{Float64, Int}, b, u,
                       Pl, Pr,
                       maxiters::Int, abstol, reltol,
                       verbose::Bool, assumptions::OperatorAssumptions)
    PREALLOCATED_UMFPACK
end

function init_cacheval(alg::UMFPACKFactorization, A, b, u, Pl, Pr, maxiters::Int, abstol,
                       reltol,
                       verbose::Bool, assumptions::OperatorAssumptions)
    A = convert(AbstractMatrix, A)

    if typeof(A) <: SparseArrays.AbstractSparseArray
        zerobased = SparseArrays.getcolptr(A)[1] == 0
        @static if VERSION < v"1.9.0-DEV.1622"
            res = SuiteSparse.UMFPACK.UmfpackLU(C_NULL, C_NULL, size(A, 1), size(A, 2),
                                                zerobased ?
                                                copy(SparseArrays.getcolptr(A)) :
                                                SuiteSparse.decrement(SparseArrays.getcolptr(A)),
                                                zerobased ? copy(rowvals(A)) :
                                                SuiteSparse.decrement(rowvals(A)),
                                                copy(nonzeros(A)), 0)
            finalizer(SuiteSparse.UMFPACK.umfpack_free_symbolic, res)
            return res
        else
            return SuiteSparse.UMFPACK.UmfpackLU(SparseMatrixCSC(size(A)..., getcolptr(A),
                                                                 rowvals(A), nonzeros(A)))
        end

    else
        @static if VERSION < v"1.9.0-DEV.1622"
            res = SuiteSparse.UMFPACK.UmfpackLU(C_NULL, C_NULL, 0, 0,
                                                [0], Int64[], eltype(A)[], 0)
            finalizer(SuiteSparse.UMFPACK.umfpack_free_symbolic, res)
            return res
        else
            return SuiteSparse.UMFPACK.UmfpackLU(SparseMatrixCSC(0, 0, [1], Int64[],
                                                                 eltype(A)[]))
        end
    end
end

function SciMLBase.solve!(cache::LinearCache, alg::UMFPACKFactorization; kwargs...)
    A = cache.A
    A = convert(AbstractMatrix, A)
    if cache.isfresh
        if alg.reuse_symbolic
            # Caches the symbolic factorization: https://github.com/JuliaLang/julia/pull/33738
            if alg.check_pattern && !(SuiteSparse.decrement(SparseArrays.getcolptr(A)) ==
                 cache.cacheval.colptr &&
                 SuiteSparse.decrement(SparseArrays.getrowval(A)) ==
                 get_cacheval(cache, :UMFPACKFactorization).rowval)
                fact = lu(SparseMatrixCSC(size(A)..., getcolptr(A), rowvals(A),
                                          nonzeros(A)))
            else
                fact = lu!(get_cacheval(cache, :UMFPACKFactorization),
                           SparseMatrixCSC(size(A)..., getcolptr(A), rowvals(A),
                                           nonzeros(A)))
            end
        else
            fact = lu(SparseMatrixCSC(size(A)..., getcolptr(A), rowvals(A), nonzeros(A)))
        end
        cache.cacheval = fact
        cache.isfresh = false
    end

    y = ldiv!(cache.u, get_cacheval(cache, :UMFPACKFactorization), cache.b)
    SciMLBase.build_linear_solution(alg, y, nothing, cache)
end

Base.@kwdef struct KLUFactorization <: AbstractFactorization
    reuse_symbolic::Bool = true
    check_pattern::Bool = true
end

const PREALLOCATED_KLU = KLU.KLUFactorization(SparseMatrixCSC(0, 0, [1], Int64[],
                                                              Float64[]))

function init_cacheval(alg::KLUFactorization,
                       A::Union{Matrix, Nothing, AbstractSciMLOperator}, b, u, Pl, Pr,
                       maxiters::Int, abstol, reltol,
                       verbose::Bool, assumptions::OperatorAssumptions)
    nothing
end

function init_cacheval(alg::KLUFactorization, A::SparseMatrixCSC{Float64, Int}, b, u, Pl,
                       Pr,
                       maxiters::Int, abstol, reltol,
                       verbose::Bool, assumptions::OperatorAssumptions)
    PREALLOCATED_KLU
end

function init_cacheval(alg::KLUFactorization, A, b, u, Pl, Pr, maxiters::Int, abstol,
                       reltol,
                       verbose::Bool, assumptions::OperatorAssumptions)
    A = convert(AbstractMatrix, A)
    if typeof(A) <: SparseArrays.AbstractSparseArray
        return KLU.KLUFactorization(SparseMatrixCSC(size(A)..., getcolptr(A), rowvals(A),
                                                    nonzeros(A)))
    else
        return KLU.KLUFactorization(SparseMatrixCSC(0, 0, [1], Int64[], eltype(A)[]))
    end
end

function SciMLBase.solve!(cache::LinearCache, alg::KLUFactorization; kwargs...)
    A = cache.A
    A = convert(AbstractMatrix, A)

    if cache.isfresh
        cacheval = get_cacheval(cache, :KLUFactorization)
        if cacheval !== nothing && alg.reuse_symbolic
            if alg.check_pattern && !(SuiteSparse.decrement(SparseArrays.getcolptr(A)) ==
                 cacheval.colptr &&
                 SuiteSparse.decrement(SparseArrays.getrowval(A)) == cacheval.rowval)
                fact = KLU.klu(SparseMatrixCSC(size(A)..., getcolptr(A), rowvals(A),
                                               nonzeros(A)))
            else
                # If we have a cacheval already, run umfpack_symbolic to ensure the symbolic factorization exists
                # This won't recompute if it does.
                KLU.klu_analyze!(cacheval)
                copyto!(cache.cacheval.nzval, nonzeros(A))
                if cache.cacheval._numeric === C_NULL # We MUST have a numeric factorization for reuse, unlike UMFPACK.
                    KLU.klu_factor!(cacheval)
                end
                fact = KLU.klu!(cacheval,
                                SparseMatrixCSC(size(A)..., getcolptr(A), rowvals(A),
                                                nonzeros(A)))
            end
        else
            # New fact each time since the sparsity pattern can change
            # and thus it needs to reallocate
            fact = KLU.klu(SparseMatrixCSC(size(A)..., getcolptr(A), rowvals(A),
                                           nonzeros(A)))
        end
        cache.cacheval = fact
        cache.isfresh = false
    end

    y = ldiv!(cache.u, get_cacheval(cache, :KLUFactorization), cache.b)
    SciMLBase.build_linear_solution(alg, y, nothing, cache)
end

## CHOLMODFactorization

Base.@kwdef struct CHOLMODFactorization{T} <: AbstractFactorization
    shift::Float64 = 0.0
    perm::T = nothing
end

const PREALLOCATED_CHOLMOD = cholesky(SparseMatrixCSC(0, 0, [1], Int64[], Float64[]))

function init_cacheval(alg::CHOLMODFactorization,
                       A::Union{Matrix, Nothing, AbstractSciMLOperator}, b, u, Pl, Pr,
                       maxiters::Int, abstol, reltol,
                       verbose::Bool, assumptions::OperatorAssumptions)
    nothing
end

function init_cacheval(alg::CHOLMODFactorization, A::SparseMatrixCSC{Float64, Int}, b, u,
                       Pl, Pr,
                       maxiters::Int, abstol, reltol,
                       verbose::Bool, assumptions::OperatorAssumptions)
    PREALLOCATED_CHOLMOD
end

function init_cacheval(alg::CHOLMODFactorization, A, b, u, Pl, Pr, maxiters::Int, abstol,
                       reltol,
                       verbose::Bool, assumptions::OperatorAssumptions)
    cholesky(SparseMatrixCSC(0, 0, [1], Int64[], eltype(A)[]))
end

function SciMLBase.solve!(cache::LinearCache, alg::CHOLMODFactorization; kwargs...)
    A = cache.A
    A = convert(AbstractMatrix, A)

    if cache.isfresh
        cacheval = get_cacheval(cache, :CHOLMODFactorization)
        fact = cholesky(A; check = false)
        if !LinearAlgebra.issuccess(fact)
            ldlt!(fact, A; check = false)
        end
        cache.cacheval = fact
        cache.isfresh = false
    end

    cache.u .= get_cacheval(cache, :CHOLMODFactorization) \ cache.b
    SciMLBase.build_linear_solution(alg, cache.u, nothing, cache)
end

## RFLUFactorization

struct RFLUFactorization{P, T} <: AbstractFactorization
    RFLUFactorization(::Val{P}, ::Val{T}) where {P, T} = new{P, T}()
end

function RFLUFactorization(; pivot = Val(true), thread = Val(true))
    RFLUFactorization(pivot, thread)
end

function init_cacheval(alg::RFLUFactorization, A, b, u, Pl, Pr, maxiters::Int,
                       abstol, reltol, verbose::Bool, assumptions::OperatorAssumptions)
    ipiv = Vector{LinearAlgebra.BlasInt}(undef, min(size(A)...))
    ArrayInterface.lu_instance(convert(AbstractMatrix, A)), ipiv
end

function init_cacheval(alg::RFLUFactorization, A::Matrix{Float64}, b, u, Pl, Pr,
                       maxiters::Int,
                       abstol, reltol, verbose::Bool, assumptions::OperatorAssumptions)
    ipiv = Vector{LinearAlgebra.BlasInt}(undef, 0)
    PREALLOCATED_LU, ipiv
end

function init_cacheval(alg::RFLUFactorization,
                       A::Union{AbstractSparseArray, AbstractSciMLOperator}, b, u, Pl, Pr,
                       maxiters::Int,
                       abstol, reltol, verbose::Bool, assumptions::OperatorAssumptions)
    nothing, nothing
end

@static if VERSION < v"1.7-"
    function init_cacheval(alg::RFLUFactorization,
                           A::Union{Diagonal, SymTridiagonal, Tridiagonal}, b, u, Pl, Pr,
                           maxiters::Int,
                           abstol, reltol, verbose::Bool, assumptions::OperatorAssumptions)
        nothing, nothing
    end
end

function SciMLBase.solve!(cache::LinearCache, alg::RFLUFactorization{P, T};
                          kwargs...) where {P, T}
    A = cache.A
    A = convert(AbstractMatrix, A)
    fact, ipiv = get_cacheval(cache, :RFLUFactorization)
    if cache.isfresh
        if length(ipiv) != min(size(A)...)
            ipiv = Vector{LinearAlgebra.BlasInt}(undef, min(size(A)...))
        end
        fact = RecursiveFactorization.lu!(A, ipiv, Val(P), Val(T))
        cache.cacheval = (fact, ipiv)
        cache.isfresh = false
    end
    y = ldiv!(cache.u, get_cacheval(cache, :RFLUFactorization)[1], cache.b)
    SciMLBase.build_linear_solution(alg, y, nothing, cache)
end

## NormalCholeskyFactorization

struct NormalCholeskyFactorization{P} <: AbstractFactorization
    pivot::P
end

function NormalCholeskyFactorization(; pivot = nothing)
    if pivot === nothing
        pivot = @static if VERSION < v"1.7beta"
            Val(true)
        else
            RowMaximum()
        end
    end
    NormalCholeskyFactorization(pivot)
end

default_alias_A(::NormalCholeskyFactorization, ::Any, ::Any) = true
default_alias_b(::NormalCholeskyFactorization, ::Any, ::Any) = true

@static if VERSION < v"1.7beta"
    normcholpivot = Val(false)
else
    normcholpivot = NoPivot()
end

const PREALLOCATED_NORMALCHOLESKY = ArrayInterface.cholesky_instance(rand(1, 1),
                                                                     normcholpivot)

function init_cacheval(alg::NormalCholeskyFactorization,
                       A::Union{AbstractSparseArray,
                                Symmetric{<:Number, <:AbstractSparseArray}}, b, u, Pl, Pr,
                       maxiters::Int, abstol, reltol, verbose::Bool,
                       assumptions::OperatorAssumptions)
    ArrayInterface.cholesky_instance(convert(AbstractMatrix, A))
end

function init_cacheval(alg::NormalCholeskyFactorization, A, b, u, Pl, Pr,
                       maxiters::Int, abstol, reltol, verbose::Bool,
                       assumptions::OperatorAssumptions)
    ArrayInterface.cholesky_instance(convert(AbstractMatrix, A), alg.pivot)
end

function init_cacheval(alg::NormalCholeskyFactorization,
                       A::Union{Diagonal, AbstractSciMLOperator}, b, u, Pl, Pr,
                       maxiters::Int, abstol, reltol, verbose::Bool,
                       assumptions::OperatorAssumptions)
    nothing
end

@static if VERSION < v"1.7-"
    function init_cacheval(alg::NormalCholeskyFactorization,
                           A::Union{Tridiagonal, SymTridiagonal}, b, u, Pl, Pr,
                           maxiters::Int, abstol, reltol, verbose::Bool,
                           assumptions::OperatorAssumptions)
        nothing
    end
end

function SciMLBase.solve!(cache::LinearCache, alg::NormalCholeskyFactorization;
                          kwargs...)
    A = cache.A
    A = convert(AbstractMatrix, A)
    if cache.isfresh
        if A isa SparseMatrixCSC
            fact = cholesky(Symmetric((A)' * A, :L))
        else
            fact = cholesky(Symmetric((A)' * A, :L), alg.pivot)
        end
        cache.cacheval = fact
        cache.isfresh = false
    end
    if A isa SparseMatrixCSC
        cache.u .= get_cacheval(cache, :NormalCholeskyFactorization) \ (A' * cache.b)
        y = cache.u
    else
        y = ldiv!(cache.u, get_cacheval(cache, :NormalCholeskyFactorization), A' * cache.b)
    end
    SciMLBase.build_linear_solution(alg, y, nothing, cache)
end

## NormalBunchKaufmanFactorization

struct NormalBunchKaufmanFactorization <: AbstractFactorization
    rook::Bool
end

function NormalBunchKaufmanFactorization(; rook = false)
    NormalBunchKaufmanFactorization(rook)
end

default_alias_A(::NormalBunchKaufmanFactorization, ::Any, ::Any) = true
default_alias_b(::NormalBunchKaufmanFactorization, ::Any, ::Any) = true

function init_cacheval(alg::NormalBunchKaufmanFactorization, A, b, u, Pl, Pr,
                       maxiters::Int, abstol, reltol, verbose::Bool,
                       assumptions::OperatorAssumptions)
    ArrayInterface.bunchkaufman_instance(convert(AbstractMatrix, A))
end

function SciMLBase.solve!(cache::LinearCache, alg::NormalBunchKaufmanFactorization;
                          kwargs...)
    A = cache.A
    A = convert(AbstractMatrix, A)
    if cache.isfresh
        fact = bunchkaufman(Symmetric((A)' * A), alg.rook)
        cache.cacheval = fact
        cache.isfresh = false
    end
    y = ldiv!(cache.u, get_cacheval(cache, :NormalBunchKaufmanFactorization), A' * cache.b)
    SciMLBase.build_linear_solution(alg, y, nothing, cache)
end

## DiagonalFactorization

struct DiagonalFactorization <: AbstractFactorization end

function init_cacheval(alg::DiagonalFactorization, A, b, u, Pl, Pr, maxiters::Int,
                       abstol, reltol, verbose::Bool, assumptions::OperatorAssumptions)
    nothing
end

function SciMLBase.solve!(cache::LinearCache, alg::DiagonalFactorization;
                          kwargs...)
    A = cache.A
    if cache.u isa Vector && cache.b isa Vector
        @simd ivdep for i in eachindex(cache.u)
            cache.u[i] = A.diag[i] \ cache.b[i]
        end
    else
        cache.u .= A.diag .\ cache.b
    end
    SciMLBase.build_linear_solution(alg, cache.u, nothing, cache)
end

## FastLAPACKFactorizations

struct WorkspaceAndFactors{W, F}
    workspace::W
    factors::F
end

# There's no options like pivot here.
# But I'm not sure it makes sense as a GenericFactorization
# since it just uses `LAPACK.getrf!`.
struct FastLUFactorization <: AbstractFactorization end

function init_cacheval(::FastLUFactorization, A, b, u, Pl, Pr,
                       maxiters::Int, abstol, reltol, verbose::Bool,
                       assumptions::OperatorAssumptions)
    ws = LUWs(A)
    return WorkspaceAndFactors(ws, ArrayInterface.lu_instance(convert(AbstractMatrix, A)))
end

function SciMLBase.solve!(cache::LinearCache, alg::FastLUFactorization; kwargs...)
    A = cache.A
    A = convert(AbstractMatrix, A)
    ws_and_fact = get_cacheval(cache, :FastLUFactorization)
    if cache.isfresh
        # we will fail here if A is a different *size* than in a previous version of the same cache.
        # it may instead be desirable to resize the workspace.
        @set! ws_and_fact.factors = LinearAlgebra.LU(LAPACK.getrf!(ws_and_fact.workspace,
                                                                   A)...)
        cache.cacheval = ws_and_fact
        cache.isfresh = false
    end
    y = ldiv!(cache.u, cache.cacheval.factors, cache.b)
    SciMLBase.build_linear_solution(alg, y, nothing, cache)
end

struct FastQRFactorization{P} <: AbstractFactorization
    pivot::P
    blocksize::Int
end

function FastQRFactorization()
    if VERSION < v"1.7beta"
        FastQRFactorization(Val(false), 36)
    else
        FastQRFactorization(NoPivot(), 36)
    end
    # is 36 or 16 better here? LinearAlgebra and FastLapackInterface use 36,
    # but QRFactorization uses 16.
end

@static if VERSION < v"1.7beta"
    function init_cacheval(alg::FastQRFactorization{Val{false}}, A, b, u, Pl, Pr,
                           maxiters::Int, abstol, reltol, verbose::Bool,
                           assumptions::OperatorAssumptions)
        ws = QRWYWs(A; blocksize = alg.blocksize)
        return WorkspaceAndFactors(ws,
                                   ArrayInterface.qr_instance(convert(AbstractMatrix, A)))
    end

    function init_cacheval(::FastQRFactorization{Val{true}}, A, b, u, Pl, Pr,
                           maxiters::Int, abstol, reltol, verbose::Bool,
                           assumptions::OperatorAssumptions)
        ws = QRpWs(A)
        return WorkspaceAndFactors(ws,
                                   ArrayInterface.qr_instance(convert(AbstractMatrix, A)))
    end
else
    function init_cacheval(alg::FastQRFactorization{NoPivot}, A, b, u, Pl, Pr,
                           maxiters::Int, abstol, reltol, verbose::Bool,
                           assumptions::OperatorAssumptions)
        ws = QRWYWs(A; blocksize = alg.blocksize)
        return WorkspaceAndFactors(ws,
                                   ArrayInterface.qr_instance(convert(AbstractMatrix, A)))
    end
    function init_cacheval(::FastQRFactorization{ColumnNorm}, A, b, u, Pl, Pr,
                           maxiters::Int, abstol, reltol, verbose::Bool,
                           assumptions::OperatorAssumptions)
        ws = QRpWs(A)
        return WorkspaceAndFactors(ws,
                                   ArrayInterface.qr_instance(convert(AbstractMatrix, A)))
    end
end

function SciMLBase.solve!(cache::LinearCache, alg::FastQRFactorization{P};
                          kwargs...) where {P}
    A = cache.A
    A = convert(AbstractMatrix, A)
    ws_and_fact = get_cacheval(cache, :FastQRFactorization)
    if cache.isfresh
        # we will fail here if A is a different *size* than in a previous version of the same cache.
        # it may instead be desirable to resize the workspace.
        nopivot = @static if VERSION < v"1.7beta"
            Val{false}
        else
            NoPivot
        end
        if P === nopivot
            @set! ws_and_fact.factors = LinearAlgebra.QRCompactWY(LAPACK.geqrt!(ws_and_fact.workspace,
                                                                                A)...)
        else
            @set! ws_and_fact.factors = LinearAlgebra.QRPivoted(LAPACK.geqp3!(ws_and_fact.workspace,
                                                                              A)...)
        end
        cache.cacheval = ws_and_fact
        cache.isfresh = false
    end
    y = ldiv!(cache.u, cache.cacheval.factors, cache.b)
    SciMLBase.build_linear_solution(alg, y, nothing, cache)
end

## SparspakFactorization is here since it's MIT licensed, not GPL

Base.@kwdef struct SparspakFactorization <: AbstractFactorization
    reuse_symbolic::Bool = true
end

const PREALLOCATED_SPARSEPAK = sparspaklu(SparseMatrixCSC(0, 0, [1], Int64[], Float64[]),
                                          factorize = false)

function init_cacheval(alg::SparspakFactorization,
                       A::Union{Matrix, Nothing, AbstractSciMLOperator}, b, u, Pl, Pr,
                       maxiters::Int, abstol, reltol,
                       verbose::Bool, assumptions::OperatorAssumptions)
    nothing
end

function init_cacheval(::SparspakFactorization, A::SparseMatrixCSC{Float64, Int}, b, u, Pl,
                       Pr, maxiters::Int, abstol,
                       reltol,
                       verbose::Bool, assumptions::OperatorAssumptions)
    PREALLOCATED_SPARSEPAK
end

function init_cacheval(::SparspakFactorization, A, b, u, Pl, Pr, maxiters::Int, abstol,
                       reltol,
                       verbose::Bool, assumptions::OperatorAssumptions)
    A = convert(AbstractMatrix, A)
    if typeof(A) <: SparseArrays.AbstractSparseArray
        return sparspaklu(SparseMatrixCSC(size(A)..., getcolptr(A), rowvals(A),
                                          nonzeros(A)),
                          factorize = false)
    else
        return sparspaklu(SparseMatrixCSC(0, 0, [1], Int64[], eltype(A)[]),
                          factorize = false)
    end
end

function SciMLBase.solve!(cache::LinearCache, alg::SparspakFactorization; kwargs...)
    A = cache.A
    if cache.isfresh
        if cache.cacheval !== nothing && alg.reuse_symbolic
            fact = sparspaklu!(get_cacheval(cache, :SparspakFactorization),
                               SparseMatrixCSC(size(A)..., getcolptr(A), rowvals(A),
                                               nonzeros(A)))
        else
            fact = sparspaklu(SparseMatrixCSC(size(A)..., getcolptr(A), rowvals(A),
                                              nonzeros(A)))
        end
        cache.cacheval = fact
        cache.isfresh = false
    end
    y = ldiv!(cache.u, get_cacheval(cache, :SparspakFactorization), cache.b)
    SciMLBase.build_linear_solution(alg, y, nothing, cache)
end
