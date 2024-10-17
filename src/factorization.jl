macro get_cacheval(cache, algsym)
    quote
        if $(esc(cache)).alg isa DefaultLinearSolver
            getfield($(esc(cache)).cacheval, $algsym)
        else
            $(esc(cache)).cacheval
        end
    end
end

_ldiv!(x, A, b) = ldiv!(x, A, b)

_ldiv!(x, A, b::SVector) = (x .= A \ b)
_ldiv!(::SVector, A, b::SVector) = (A \ b)
_ldiv!(::SVector, A, b) = (A \ b)

function _ldiv!(x::Vector, A::Factorization, b::Vector)
    # workaround https://github.com/JuliaLang/julia/issues/43507
    # Fallback if working with non-square matrices
    length(x) != length(b) && return ldiv!(x, A, b)
    copyto!(x, b)
    ldiv!(A, x)
end

# RF Bad fallback: will fail if `A` is just a stand-in
# This should instead just create the factorization type.
function init_cacheval(alg::AbstractFactorization, A, b, u, Pl, Pr, maxiters::Int, abstol,
        reltol, verbose::Bool, assumptions::OperatorAssumptions)
    do_factorization(alg, convert(AbstractMatrix, A), b, u)
end

## LU Factorizations

"""
`LUFactorization(pivot=LinearAlgebra.RowMaximum())`

Julia's built in `lu`. Equivalent to calling `lu!(A)`

  - On dense matrices, this uses the current BLAS implementation of the user's computer,
    which by default is OpenBLAS but will use MKL if the user does `using MKL` in their
    system.
  - On sparse matrices, this will use UMFPACK from SparseArrays. Note that this will not
    cache the symbolic factorization.
  - On CuMatrix, it will use a CUDA-accelerated LU from CuSolver.
  - On BandedMatrix and BlockBandedMatrix, it will use a banded LU.

## Positional Arguments

  - pivot: The choice of pivoting. Defaults to `LinearAlgebra.RowMaximum()`. The other choice is
    `LinearAlgebra.NoPivot()`.
"""
Base.@kwdef struct LUFactorization{P} <: AbstractDenseFactorization
    pivot::P = LinearAlgebra.RowMaximum()
    reuse_symbolic::Bool = true
    check_pattern::Bool = true # Check factorization re-use
end

# Legacy dispatch
LUFactorization(pivot) = LUFactorization(; pivot = RowMaximum())

"""
`GenericLUFactorization(pivot=LinearAlgebra.RowMaximum())`

Julia's built in generic LU factorization. Equivalent to calling LinearAlgebra.generic_lufact!.
Supports arbitrary number types but does not achieve as good scaling as BLAS-based LU implementations.
Has low overhead and is good for small matrices.

## Positional Arguments

  - pivot: The choice of pivoting. Defaults to `LinearAlgebra.RowMaximum()`. The other choice is
    `LinearAlgebra.NoPivot()`.
"""
struct GenericLUFactorization{P} <: AbstractDenseFactorization
    pivot::P
end

GenericLUFactorization() = GenericLUFactorization(RowMaximum())

function SciMLBase.solve!(cache::LinearCache, alg::LUFactorization; kwargs...)
    A = cache.A
    A = convert(AbstractMatrix, A)
    if cache.isfresh
        cacheval = @get_cacheval(cache, :LUFactorization)
        if A isa AbstractSparseMatrix && alg.reuse_symbolic
            # Caches the symbolic factorization: https://github.com/JuliaLang/julia/pull/33738
            # If SparseMatrixCSC, check if the pattern has changed
            if alg.check_pattern && pattern_changed(cacheval, A)
                fact = lu(A, check = false)
            else
                fact = lu!(cacheval, A, check = false)
            end
        else
            fact = lu(A, check = false)
        end
        cache.cacheval = fact

        if !LinearAlgebra.issuccess(fact)
            return SciMLBase.build_linear_solution(
                alg, cache.u, nothing, cache; retcode = ReturnCode.Failure)
        end

        cache.isfresh = false
    end

    F = @get_cacheval(cache, :LUFactorization)
    y = _ldiv!(cache.u, F, cache.b)
    SciMLBase.build_linear_solution(alg, y, nothing, cache)
end

function do_factorization(alg::LUFactorization, A, b, u)
    A = convert(AbstractMatrix, A)
    if A isa AbstractSparseMatrixCSC
        return lu(SparseMatrixCSC(size(A)..., getcolptr(A), rowvals(A), nonzeros(A)),
            check = false)
    elseif A isa GPUArraysCore.AnyGPUArray
        fact = lu(A; check = false)
    elseif !ArrayInterface.can_setindex(typeof(A))
        fact = lu(A, alg.pivot, check = false)
    else
        fact = lu!(A, alg.pivot, check = false)
    end
    return fact
end

function do_factorization(alg::GenericLUFactorization, A, b, u)
    A = convert(AbstractMatrix, A)
    fact = LinearAlgebra.generic_lufact!(A, alg.pivot, check = false)
    return fact
end

function init_cacheval(
        alg::Union{LUFactorization, GenericLUFactorization}, A, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Bool,
        assumptions::OperatorAssumptions)
    ArrayInterface.lu_instance(convert(AbstractMatrix, A))
end

function init_cacheval(alg::Union{LUFactorization, GenericLUFactorization},
        A::Union{<:Adjoint, <:Transpose}, b, u, Pl, Pr, maxiters::Int, abstol, reltol,
        verbose::Bool, assumptions::OperatorAssumptions)
    error_no_cudss_lu(A)
    if alg isa LUFactorization
        return lu(A; check = false)
    else
        A isa GPUArraysCore.AnyGPUArray && return nothing
        return LinearAlgebra.generic_lufact!(copy(A), alg.pivot; check = false)
    end
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

## QRFactorization

"""
`QRFactorization(pivot=LinearAlgebra.NoPivot(),blocksize=16)`

Julia's built in `qr`. Equivalent to calling `qr!(A)`.

  - On dense matrices, this uses the current BLAS implementation of the user's computer
    which by default is OpenBLAS but will use MKL if the user does `using MKL` in their
    system.
  - On sparse matrices, this will use SPQR from SparseArrays
  - On CuMatrix, it will use a CUDA-accelerated QR from CuSolver.
  - On BandedMatrix and BlockBandedMatrix, it will use a banded QR.
"""
struct QRFactorization{P} <: AbstractDenseFactorization
    pivot::P
    blocksize::Int
    inplace::Bool
end

QRFactorization(inplace = true) = QRFactorization(NoPivot(), 16, inplace)

function QRFactorization(pivot::LinearAlgebra.PivotingStrategy, inplace::Bool = true)
    QRFactorization(pivot, 16, inplace)
end

function do_factorization(alg::QRFactorization, A, b, u)
    A = convert(AbstractMatrix, A)
    if ArrayInterface.can_setindex(typeof(A))
        if alg.inplace && !(A isa SparseMatrixCSC) && !(A isa GPUArraysCore.AnyGPUArray)
            if A isa Symmetric
                fact = qr(A, alg.pivot)
            else
                fact = qr!(A, alg.pivot)
            end
        else
            fact = qr(A) # CUDA.jl does not allow other args!
        end
    else
        fact = qr(A, alg.pivot)
    end
    return fact
end

function init_cacheval(alg::QRFactorization, A, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Bool,
        assumptions::OperatorAssumptions)
    ArrayInterface.qr_instance(convert(AbstractMatrix, A), alg.pivot)
end

function init_cacheval(alg::QRFactorization, A::Symmetric, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Bool,
        assumptions::OperatorAssumptions)
    return qr(convert(AbstractMatrix, A), alg.pivot)
end

const PREALLOCATED_QR_ColumnNorm = ArrayInterface.qr_instance(rand(1, 1), ColumnNorm())

function init_cacheval(alg::QRFactorization{ColumnNorm}, A::Matrix{Float64}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Bool, assumptions::OperatorAssumptions)
    return PREALLOCATED_QR_ColumnNorm
end

function init_cacheval(
        alg::QRFactorization, A::Union{<:Adjoint, <:Transpose}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Bool, assumptions::OperatorAssumptions)
    A isa GPUArraysCore.AnyGPUArray && return qr(A)
    return qr(A, alg.pivot)
end

const PREALLOCATED_QR_NoPivot = ArrayInterface.qr_instance(rand(1, 1))

function init_cacheval(alg::QRFactorization{NoPivot}, A::Matrix{Float64}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Bool, assumptions::OperatorAssumptions)
    return PREALLOCATED_QR_NoPivot
end

function init_cacheval(alg::QRFactorization, A::AbstractSciMLOperator, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Bool,
        assumptions::OperatorAssumptions)
    nothing
end

## CholeskyFactorization

"""
`CholeskyFactorization(; pivot = nothing, tol = 0.0, shift = 0.0, perm = nothing)`

Julia's built in `cholesky`. Equivalent to calling `cholesky!(A)`.

## Keyword Arguments

  - pivot: defaluts to NoPivot, can also be RowMaximum.
  - tol: the tol argument in CHOLMOD. Only used for sparse matrices.
  - shift: the shift argument in CHOLMOD. Only used for sparse matrices.
  - perm: the perm argument in CHOLMOD. Only used for sparse matrices.
"""
struct CholeskyFactorization{P, P2} <: AbstractDenseFactorization
    pivot::P
    tol::Int
    shift::Float64
    perm::P2
end

function CholeskyFactorization(; pivot = nothing, tol = 0.0, shift = 0.0, perm = nothing)
    pivot === nothing && (pivot = NoPivot())
    CholeskyFactorization(pivot, 16, shift, perm)
end

function do_factorization(alg::CholeskyFactorization, A, b, u)
    A = convert(AbstractMatrix, A)
    if A isa SparseMatrixCSC
        fact = cholesky(A; shift = alg.shift, check = false, perm = alg.perm)
    elseif A isa GPUArraysCore.AnyGPUArray
        fact = cholesky(A; check = false)
    elseif alg.pivot === Val(false) || alg.pivot === NoPivot()
        fact = cholesky!(A, alg.pivot; check = false)
    else
        fact = cholesky!(A, alg.pivot; tol = alg.tol, check = false)
    end
    return fact
end

function init_cacheval(alg::CholeskyFactorization, A::SMatrix{S1, S2}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Bool,
        assumptions::OperatorAssumptions) where {S1, S2}
    cholesky(A)
end

function init_cacheval(alg::CholeskyFactorization, A::GPUArraysCore.AnyGPUArray, b, u, Pl,
        Pr, maxiters::Int, abstol, reltol, verbose::Bool, assumptions::OperatorAssumptions)
    cholesky(A; check = false)
end

function init_cacheval(alg::CholeskyFactorization, A, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Bool, assumptions::OperatorAssumptions)
    ArrayInterface.cholesky_instance(convert(AbstractMatrix, A), alg.pivot)
end

const PREALLOCATED_CHOLESKY = ArrayInterface.cholesky_instance(rand(1, 1), NoPivot())

function init_cacheval(alg::CholeskyFactorization, A::Matrix{Float64}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Bool,
        assumptions::OperatorAssumptions)
    PREALLOCATED_CHOLESKY
end

function init_cacheval(alg::CholeskyFactorization,
        A::Union{Diagonal, AbstractSciMLOperator}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Bool,
        assumptions::OperatorAssumptions)
    nothing
end

## LDLtFactorization

struct LDLtFactorization{T} <: AbstractDenseFactorization
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

"""
`SVDFactorization(full=false,alg=LinearAlgebra.DivideAndConquer())`

Julia's built in `svd`. Equivalent to `svd!(A)`.

  - On dense matrices, this uses the current BLAS implementation of the user's computer
    which by default is OpenBLAS but will use MKL if the user does `using MKL` in their
    system.
"""
struct SVDFactorization{A} <: AbstractDenseFactorization
    full::Bool
    alg::A
end

SVDFactorization() = SVDFactorization(false, LinearAlgebra.DivideAndConquer())

function do_factorization(alg::SVDFactorization, A, b, u)
    A = convert(AbstractMatrix, A)
    if ArrayInterface.can_setindex(typeof(A))
        fact = svd!(A; alg.full, alg.alg)
    else
        fact = svd(A; alg.full)
    end
    return fact
end

function init_cacheval(alg::SVDFactorization, A::Union{Matrix, SMatrix}, b, u, Pl, Pr,
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

## BunchKaufmanFactorization

"""
`BunchKaufmanFactorization(; rook = false)`

Julia's built in `bunchkaufman`. Equivalent to calling `bunchkaufman(A)`.
Only for Symmetric matrices.

## Keyword Arguments

  - rook: whether to perform rook pivoting. Defaults to false.
"""
Base.@kwdef struct BunchKaufmanFactorization <: AbstractDenseFactorization
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

const PREALLOCATED_BUNCHKAUFMAN = ArrayInterface.bunchkaufman_instance(Symmetric(rand(1,
    1)))

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

"""
`GenericFactorization(;fact_alg=LinearAlgebra.factorize)`: Constructs a linear solver from a generic
factorization algorithm `fact_alg` which complies with the Base.LinearAlgebra
factorization API. Quoting from Base:

      * If `A` is upper or lower triangular (or diagonal), no factorization of `A` is
        required. The system is then solved with either forward or backward substitution.
        For non-triangular square matrices, an LU factorization is used.
        For rectangular `A` the result is the minimum-norm least squares solution computed by a
        pivoted QR factorization of `A` and a rank estimate of `A` based on the R factor.
        When `A` is sparse, a similar polyalgorithm is used. For indefinite matrices, the `LDLt`
        factorization does not use pivoting during the numerical factorization and therefore the
        procedure can fail even for invertible matrices.

## Keyword Arguments

  - fact_alg: the factorization algorithm to use. Defaults to `LinearAlgebra.factorize`, but can be
    swapped to choices like `lu`, `qr`
"""
struct GenericFactorization{F} <: AbstractDenseFactorization
    fact_alg::F
end

GenericFactorization(; fact_alg = LinearAlgebra.factorize) = GenericFactorization(fact_alg)

function do_factorization(alg::GenericFactorization, A, b, u)
    A = convert(AbstractMatrix, A)
    fact = alg.fact_alg(A)
    return fact
end

function init_cacheval(
        alg::GenericFactorization{typeof(lu)}, A::AbstractMatrix, b, u, Pl, Pr,
        maxiters::Int,
        abstol, reltol, verbose::Bool, assumptions::OperatorAssumptions)
    ArrayInterface.lu_instance(A)
end
function init_cacheval(
        alg::GenericFactorization{typeof(lu!)}, A::AbstractMatrix, b, u, Pl, Pr,
        maxiters::Int,
        abstol, reltol, verbose::Bool, assumptions::OperatorAssumptions)
    ArrayInterface.lu_instance(A)
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
function init_cacheval(
        alg::GenericFactorization{typeof(lu!)}, A::Tridiagonal, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Bool,
        assumptions::OperatorAssumptions)
    ArrayInterface.lu_instance(A)
end
function init_cacheval(
        alg::GenericFactorization{typeof(lu!)}, A::SymTridiagonal{T, V}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Bool,
        assumptions::OperatorAssumptions) where {T, V}
    LinearAlgebra.LDLt{T, SymTridiagonal{T, V}}(A)
end
function init_cacheval(
        alg::GenericFactorization{typeof(lu)}, A::SymTridiagonal{T, V}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Bool,
        assumptions::OperatorAssumptions) where {T, V}
    LinearAlgebra.LDLt{T, SymTridiagonal{T, V}}(A)
end

function init_cacheval(
        alg::GenericFactorization{typeof(qr)}, A::AbstractMatrix, b, u, Pl, Pr,
        maxiters::Int,
        abstol, reltol, verbose::Bool, assumptions::OperatorAssumptions)
    ArrayInterface.qr_instance(A)
end
function init_cacheval(
        alg::GenericFactorization{typeof(qr!)}, A::AbstractMatrix, b, u, Pl, Pr,
        maxiters::Int,
        abstol, reltol, verbose::Bool, assumptions::OperatorAssumptions)
    ArrayInterface.qr_instance(A)
end
function init_cacheval(
        alg::GenericFactorization{typeof(qr)}, A::SymTridiagonal{T, V}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Bool,
        assumptions::OperatorAssumptions) where {T, V}
    LinearAlgebra.LDLt{T, SymTridiagonal{T, V}}(A)
end
function init_cacheval(
        alg::GenericFactorization{typeof(qr!)}, A::SymTridiagonal{T, V}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Bool,
        assumptions::OperatorAssumptions) where {T, V}
    LinearAlgebra.LDLt{T, SymTridiagonal{T, V}}(A)
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
function init_cacheval(
        alg::GenericFactorization{typeof(qr!)}, A::Tridiagonal, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Bool,
        assumptions::OperatorAssumptions)
    ArrayInterface.qr_instance(A)
end

function init_cacheval(
        alg::GenericFactorization{typeof(svd)}, A::AbstractMatrix, b, u, Pl, Pr,
        maxiters::Int,
        abstol, reltol, verbose::Bool, assumptions::OperatorAssumptions)
    ArrayInterface.svd_instance(A)
end
function init_cacheval(
        alg::GenericFactorization{typeof(svd!)}, A::AbstractMatrix, b, u, Pl, Pr,
        maxiters::Int,
        abstol, reltol, verbose::Bool, assumptions::OperatorAssumptions)
    ArrayInterface.svd_instance(A)
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
function init_cacheval(
        alg::GenericFactorization{typeof(svd)}, A::Tridiagonal, b, u, Pl, Pr,
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
function init_cacheval(
        alg::GenericFactorization{typeof(svd!)}, A::SymTridiagonal{T, V}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Bool,
        assumptions::OperatorAssumptions) where {T, V}
    LinearAlgebra.LDLt{T, SymTridiagonal{T, V}}(A)
end
function init_cacheval(
        alg::GenericFactorization{typeof(svd)}, A::SymTridiagonal{T, V}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Bool,
        assumptions::OperatorAssumptions) where {T, V}
    LinearAlgebra.LDLt{T, SymTridiagonal{T, V}}(A)
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
function init_cacheval(alg::GenericFactorization, A, b, u, Pl, Pr,
        maxiters::Int,
        abstol, reltol, verbose::Bool, assumptions::OperatorAssumptions)
    init_cacheval(alg, convert(AbstractMatrix, A), b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Bool,
        assumptions::OperatorAssumptions)
end
function init_cacheval(alg::GenericFactorization, A::AbstractMatrix, b, u, Pl, Pr,
        maxiters::Int,
        abstol, reltol, verbose::Bool, assumptions::OperatorAssumptions)
    do_factorization(alg, A, b, u)
end

function init_cacheval(
        alg::Union{GenericFactorization{typeof(bunchkaufman!)},
            GenericFactorization{typeof(bunchkaufman)}},
        A::Union{Hermitian, Symmetric}, b, u, Pl, Pr, maxiters::Int, abstol,
        reltol, verbose::Bool, assumptions::OperatorAssumptions)
    BunchKaufman(A.data, Array(1:size(A, 1)), A.uplo, true, false, 0)
end

function init_cacheval(
        alg::Union{GenericFactorization{typeof(bunchkaufman!)},
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
function init_cacheval(
        alg::GenericFactorization{typeof(cholesky)}, A::AbstractMatrix, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Bool,
        assumptions::OperatorAssumptions)
    newA = copy(convert(AbstractMatrix, A))
    do_factorization(alg, newA, b, u)
end
function init_cacheval(
        alg::GenericFactorization{typeof(cholesky!)}, A::AbstractMatrix, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Bool,
        assumptions::OperatorAssumptions)
    newA = copy(convert(AbstractMatrix, A))
    do_factorization(alg, newA, b, u)
end
function init_cacheval(alg::GenericFactorization{typeof(cholesky!)},
        A::Diagonal, b, u, Pl, Pr, maxiters::Int,
        abstol, reltol, verbose::Bool, assumptions::OperatorAssumptions)
    Diagonal(inv.(A.diag))
end
function init_cacheval(
        alg::GenericFactorization{typeof(cholesky!)}, A::Tridiagonal, b, u, Pl, Pr,
        maxiters::Int,
        abstol, reltol, verbose::Bool, assumptions::OperatorAssumptions)
    ArrayInterface.lu_instance(A)
end
function init_cacheval(
        alg::GenericFactorization{typeof(cholesky!)}, A::SymTridiagonal{T, V}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Bool,
        assumptions::OperatorAssumptions) where {T, V}
    LinearAlgebra.LDLt{T, SymTridiagonal{T, V}}(A)
end
function init_cacheval(alg::GenericFactorization{typeof(cholesky)},
        A::Diagonal, b, u, Pl, Pr, maxiters::Int,
        abstol, reltol, verbose::Bool, assumptions::OperatorAssumptions)
    Diagonal(inv.(A.diag))
end
function init_cacheval(
        alg::GenericFactorization{typeof(cholesky)}, A::Tridiagonal, b, u, Pl, Pr,
        maxiters::Int,
        abstol, reltol, verbose::Bool, assumptions::OperatorAssumptions)
    ArrayInterface.lu_instance(A)
end
function init_cacheval(
        alg::GenericFactorization{typeof(cholesky)}, A::SymTridiagonal{T, V}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Bool,
        assumptions::OperatorAssumptions) where {T, V}
    LinearAlgebra.LDLt{T, SymTridiagonal{T, V}}(A)
end

function init_cacheval(alg::GenericFactorization,
        A::Union{Hermitian{T, <:SparseMatrixCSC},
            Symmetric{T, <:SparseMatrixCSC}}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Bool,
        assumptions::OperatorAssumptions) where {T}
    newA = copy(convert(AbstractMatrix, A))
    do_factorization(alg, newA, b, u)
end

################################## Factorizations which require solve! overloads

## RFLUFactorization

"""
`RFLUFactorization()`

A fast pure Julia LU-factorization implementation
using RecursiveFactorization.jl. This is by far the fastest LU-factorization
implementation, usually outperforming OpenBLAS and MKL for smaller matrices
(<500x500), but currently optimized only for Base `Array` with `Float32` or `Float64`.
Additional optimization for complex matrices is in the works.
"""
struct RFLUFactorization{P, T} <: AbstractDenseFactorization
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

function init_cacheval(alg::RFLUFactorization,
        A::Union{Diagonal, SymTridiagonal, Tridiagonal}, b, u, Pl, Pr,
        maxiters::Int,
        abstol, reltol, verbose::Bool, assumptions::OperatorAssumptions)
    nothing, nothing
end

function SciMLBase.solve!(cache::LinearCache, alg::RFLUFactorization{P, T};
        kwargs...) where {P, T}
    A = cache.A
    A = convert(AbstractMatrix, A)
    fact, ipiv = @get_cacheval(cache, :RFLUFactorization)
    if cache.isfresh
        if length(ipiv) != min(size(A)...)
            ipiv = Vector{LinearAlgebra.BlasInt}(undef, min(size(A)...))
        end
        fact = RecursiveFactorization.lu!(A, ipiv, Val(P), Val(T), check = false)
        cache.cacheval = (fact, ipiv)

        if !LinearAlgebra.issuccess(fact)
            return SciMLBase.build_linear_solution(
                alg, cache.u, nothing, cache; retcode = ReturnCode.Failure)
        end

        cache.isfresh = false
    end
    y = ldiv!(cache.u, @get_cacheval(cache, :RFLUFactorization)[1], cache.b)
    SciMLBase.build_linear_solution(alg, y, nothing, cache)
end

## NormalCholeskyFactorization

"""
`NormalCholeskyFactorization(pivot = RowMaximum())`

A fast factorization which uses a Cholesky factorization on A * A'. Can be much
faster than LU factorization, but is not as numerically stable and thus should only
be applied to well-conditioned matrices.

## Positional Arguments

  - pivot: Defaults to RowMaximum(), but can be NoPivot()
"""
struct NormalCholeskyFactorization{P} <: AbstractDenseFactorization
    pivot::P
end

function NormalCholeskyFactorization(; pivot = nothing)
    pivot === nothing && (pivot = NoPivot())
    NormalCholeskyFactorization(pivot)
end

default_alias_A(::NormalCholeskyFactorization, ::Any, ::Any) = true
default_alias_b(::NormalCholeskyFactorization, ::Any, ::Any) = true

const PREALLOCATED_NORMALCHOLESKY = ArrayInterface.cholesky_instance(rand(1, 1), NoPivot())

function init_cacheval(alg::NormalCholeskyFactorization,
        A::Union{AbstractSparseArray, GPUArraysCore.AnyGPUArray,
            Symmetric{<:Number, <:AbstractSparseArray}}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Bool,
        assumptions::OperatorAssumptions)
    ArrayInterface.cholesky_instance(convert(AbstractMatrix, A))
end

function init_cacheval(alg::NormalCholeskyFactorization, A::SMatrix, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Bool,
        assumptions::OperatorAssumptions)
    return cholesky(Symmetric((A)' * A))
end

function init_cacheval(alg::NormalCholeskyFactorization, A, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Bool,
        assumptions::OperatorAssumptions)
    A_ = convert(AbstractMatrix, A)
    return ArrayInterface.cholesky_instance(
        Symmetric(Matrix{eltype(A)}(undef, 0, 0)), alg.pivot)
end

const PREALLOCATED_NORMALCHOLESKY_SYMMETRIC = ArrayInterface.cholesky_instance(
    Symmetric(rand(1, 1)), NoPivot())

function init_cacheval(alg::NormalCholeskyFactorization, A::Matrix{Float64}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Bool, assumptions::OperatorAssumptions)
    return PREALLOCATED_NORMALCHOLESKY_SYMMETRIC
end

function init_cacheval(alg::NormalCholeskyFactorization,
        A::Union{Diagonal, AbstractSciMLOperator}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Bool,
        assumptions::OperatorAssumptions)
    nothing
end

function SciMLBase.solve!(cache::LinearCache, alg::NormalCholeskyFactorization; kwargs...)
    A = cache.A
    A = convert(AbstractMatrix, A)
    if cache.isfresh
        if A isa SparseMatrixCSC || A isa GPUArraysCore.AnyGPUArray || A isa SMatrix
            fact = cholesky(Symmetric((A)' * A); check = false)
        else
            fact = cholesky(Symmetric((A)' * A), alg.pivot; check = false)
        end
        cache.cacheval = fact
        cache.isfresh = false
    end
    if A isa SparseMatrixCSC
        cache.u .= @get_cacheval(cache, :NormalCholeskyFactorization) \ (A' * cache.b)
        y = cache.u
    elseif A isa StaticArray
        cache.u = @get_cacheval(cache, :NormalCholeskyFactorization) \ (A' * cache.b)
        y = cache.u
    else
        y = ldiv!(cache.u, @get_cacheval(cache, :NormalCholeskyFactorization), A' * cache.b)
    end
    SciMLBase.build_linear_solution(alg, y, nothing, cache)
end

## NormalBunchKaufmanFactorization

"""
`NormalBunchKaufmanFactorization(rook = false)`

A fast factorization which uses a BunchKaufman factorization on A * A'. Can be much
faster than LU factorization, but is not as numerically stable and thus should only
be applied to well-conditioned matrices.

## Positional Arguments

  - rook: whether to perform rook pivoting. Defaults to false.
"""
struct NormalBunchKaufmanFactorization <: AbstractDenseFactorization
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
    y = ldiv!(cache.u, @get_cacheval(cache, :NormalBunchKaufmanFactorization), A' * cache.b)
    SciMLBase.build_linear_solution(alg, y, nothing, cache)
end

## DiagonalFactorization

"""
`DiagonalFactorization()`

A special implementation only for solving `Diagonal` matrices fast.
"""
struct DiagonalFactorization <: AbstractDenseFactorization end

function init_cacheval(alg::DiagonalFactorization, A, b, u, Pl, Pr, maxiters::Int,
        abstol, reltol, verbose::Bool, assumptions::OperatorAssumptions)
    nothing
end

function SciMLBase.solve!(cache::LinearCache, alg::DiagonalFactorization;
        kwargs...)
    A = convert(AbstractMatrix, cache.A)
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
"""
`FastLUFactorization()`

The FastLapackInterface.jl version of the LU factorization. Notably,
this version does not allow for choice of pivoting method.
"""
struct FastLUFactorization <: AbstractDenseFactorization end

function init_cacheval(::FastLUFactorization, A, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Bool,
        assumptions::OperatorAssumptions)
    ws = LUWs(A)
    return WorkspaceAndFactors(ws, ArrayInterface.lu_instance(convert(AbstractMatrix, A)))
end

function SciMLBase.solve!(cache::LinearCache, alg::FastLUFactorization; kwargs...)
    A = cache.A
    A = convert(AbstractMatrix, A)
    ws_and_fact = @get_cacheval(cache, :FastLUFactorization)
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

"""
`FastQRFactorization()`

The FastLapackInterface.jl version of the QR factorization.
"""
struct FastQRFactorization{P} <: AbstractDenseFactorization
    pivot::P
    blocksize::Int
end

# is 36 or 16 better here? LinearAlgebra and FastLapackInterface use 36,
# but QRFactorization uses 16.
FastQRFactorization() = FastQRFactorization(NoPivot(), 36)

function init_cacheval(alg::FastQRFactorization{NoPivot}, A::AbstractMatrix, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Bool,
        assumptions::OperatorAssumptions)
    ws = QRWYWs(A; blocksize = alg.blocksize)
    return WorkspaceAndFactors(ws,
        ArrayInterface.qr_instance(convert(AbstractMatrix, A)))
end
function init_cacheval(::FastQRFactorization{ColumnNorm}, A::AbstractMatrix, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Bool,
        assumptions::OperatorAssumptions)
    ws = QRpWs(A)
    return WorkspaceAndFactors(ws,
        ArrayInterface.qr_instance(convert(AbstractMatrix, A)))
end

function init_cacheval(alg::FastQRFactorization, A, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Bool,
        assumptions::OperatorAssumptions)
    return init_cacheval(alg, convert(AbstractMatrix, A), b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Bool,
        assumptions::OperatorAssumptions)
end

function SciMLBase.solve!(cache::LinearCache, alg::FastQRFactorization{P};
        kwargs...) where {P}
    A = cache.A
    A = convert(AbstractMatrix, A)
    ws_and_fact = @get_cacheval(cache, :FastQRFactorization)
    if cache.isfresh
        # we will fail here if A is a different *size* than in a previous version of the same cache.
        # it may instead be desirable to resize the workspace.
        if P === NoPivot
            @set! ws_and_fact.factors = LinearAlgebra.QRCompactWY(LAPACK.geqrt!(
                ws_and_fact.workspace,
                A)...)
        else
            @set! ws_and_fact.factors = LinearAlgebra.QRPivoted(LAPACK.geqp3!(
                ws_and_fact.workspace,
                A)...)
        end
        cache.cacheval = ws_and_fact
        cache.isfresh = false
    end
    y = ldiv!(cache.u, cache.cacheval.factors, cache.b)
    SciMLBase.build_linear_solution(alg, y, nothing, cache)
end

## SparspakFactorization is here since it's MIT licensed, not GPL

"""
`SparspakFactorization(reuse_symbolic = true)`

This is the translation of the well-known sparse matrix software Sparspak
(Waterloo Sparse Matrix Package), solving
large sparse systems of linear algebraic equations. Sparspak is composed of the
subroutines from the book "Computer Solution of Large Sparse Positive Definite
Systems" by Alan George and Joseph Liu. Originally written in Fortran 77, later
rewritten in Fortran 90. Here is the software translated into Julia.

The Julia rewrite is released  under the MIT license with an express permission
from the authors of the Fortran package. The package uses multiple
dispatch to route around standard BLAS routines in the case e.g. of arbitrary-precision
floating point numbers or ForwardDiff.Dual.
This e.g. allows for Automatic Differentiation (AD) of a sparse-matrix solve.
"""
Base.@kwdef struct SparspakFactorization <: AbstractSparseFactorization
    reuse_symbolic::Bool = true
end

const PREALLOCATED_SPARSEPAK = sparspaklu(SparseMatrixCSC(0, 0, [1], Int[], Float64[]),
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
    if A isa SparseArrays.AbstractSparseArray
        return sparspaklu(
            SparseMatrixCSC(size(A)..., getcolptr(A), rowvals(A),
                nonzeros(A)),
            factorize = false)
    else
        return sparspaklu(SparseMatrixCSC(0, 0, [1], Int[], eltype(A)[]),
            factorize = false)
    end
end

function init_cacheval(::SparspakFactorization, ::StaticArray, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Bool, assumptions::OperatorAssumptions)
    nothing
end

function SciMLBase.solve!(cache::LinearCache, alg::SparspakFactorization; kwargs...)
    A = cache.A
    if cache.isfresh
        if cache.cacheval !== nothing && alg.reuse_symbolic
            fact = sparspaklu!(@get_cacheval(cache, :SparspakFactorization),
                SparseMatrixCSC(size(A)..., getcolptr(A), rowvals(A),
                    nonzeros(A)))
        else
            fact = sparspaklu(SparseMatrixCSC(size(A)..., getcolptr(A), rowvals(A),
                nonzeros(A)))
        end
        cache.cacheval = fact
        cache.isfresh = false
    end
    y = ldiv!(cache.u, @get_cacheval(cache, :SparspakFactorization), cache.b)
    SciMLBase.build_linear_solution(alg, y, nothing, cache)
end

for alg in vcat(InteractiveUtils.subtypes(AbstractDenseFactorization),
    InteractiveUtils.subtypes(AbstractSparseFactorization))
    @eval function init_cacheval(alg::$alg, A::MatrixOperator, b, u, Pl, Pr,
            maxiters::Int, abstol, reltol, verbose::Bool,
            assumptions::OperatorAssumptions)
        init_cacheval(alg, A.A, b, u, Pl, Pr,
            maxiters::Int, abstol, reltol, verbose::Bool,
            assumptions::OperatorAssumptions)
    end
end
