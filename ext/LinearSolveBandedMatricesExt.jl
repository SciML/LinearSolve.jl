module LinearSolveBandedMatricesExt

using BandedMatrices, LinearAlgebra, LinearSolve
import LinearSolve: defaultalg,
    do_factorization, init_cacheval, DefaultLinearSolver,
    DefaultAlgorithmChoice, LinearVerbosity

# Defaults for BandedMatrices
function defaultalg(A::BandedMatrix, b, oa::OperatorAssumptions{Bool})
    if oa.issq
        return DefaultLinearSolver(DefaultAlgorithmChoice.LUFactorization)
    elseif LinearSolve.is_underdetermined(A)
        error("No solver for underdetermined `A::BandedMatrix` is currently implemented!")
    else
        return DefaultLinearSolver(DefaultAlgorithmChoice.QRFactorization)
    end
end

function defaultalg(
        A::BandedMatrix{T}, b, oa::OperatorAssumptions{Bool}
    ) where {T <: BigFloat}
    return DefaultLinearSolver(DefaultAlgorithmChoice.QRFactorization)
end

function defaultalg(A::Symmetric{<:Number, <:BandedMatrix}, b, ::OperatorAssumptions{Bool})
    return DefaultLinearSolver(DefaultAlgorithmChoice.CholeskyFactorization)
end

# BandedMatrices `qr` doesn't support column pivoting, so convert to dense when
# pivoting is requested (e.g. ColumnNorm fallback from singular LU).
function do_factorization(alg::QRFactorization, A::BandedMatrix, b, u)
    if alg.pivot isa NoPivot
        return alg.inplace ? qr!(A) : qr(A)
    else
        return qr!(Matrix(A), alg.pivot)
    end
end

function do_factorization(alg::LUFactorization, A::BandedMatrix, b, u)
    # BandedMatrices.jl requires Val-based pivot argument for lu!
    _pivot = alg.pivot isa NoPivot ? Val(false) : Val(true)
    return lu!(A, _pivot; check = false)
end

# For BandedMatrix
for alg in (
        :SVDFactorization, :MKLLUFactorization, :DiagonalFactorization,
        :SparspakFactorization, :KLUFactorization, :UMFPACKFactorization,
        :GenericLUFactorization, :RFLUFactorization, :BunchKaufmanFactorization,
        :CHOLMODFactorization, :NormalCholeskyFactorization, :LDLtFactorization,
        :AppleAccelerateLUFactorization, :CholeskyFactorization,
    )
    @eval begin
        function init_cacheval(
                ::$(alg), ::BandedMatrix, b, u, Pl, Pr, maxiters::Int,
                abstol, reltol, verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
            )
            return nothing
        end
    end
end

function init_cacheval(
        ::LUFactorization, A::BandedMatrix{T}, b, u, Pl, Pr, maxiters::Int,
        abstol, reltol, verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    ) where {T}
    (T <: BigFloat) && return qr(similar(A, 0, 0))
    return lu(similar(A, 0, 0))
end

# Column-pivoted QR on BandedMatrix converts to dense, so cache a dense QRPivoted
function init_cacheval(
        ::QRFactorization{ColumnNorm}, A::BandedMatrix, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    )
    return LinearAlgebra.qr(Matrix{eltype(A)}(undef, 0, 0), ColumnNorm())
end

# For Symmetric BandedMatrix
for alg in (
        :SVDFactorization, :MKLLUFactorization, :DiagonalFactorization,
        :SparspakFactorization, :KLUFactorization, :UMFPACKFactorization,
        :GenericLUFactorization, :RFLUFactorization, :BunchKaufmanFactorization,
        :CHOLMODFactorization, :NormalCholeskyFactorization,
        :AppleAccelerateLUFactorization, :QRFactorization, :LUFactorization,
    )
    @eval begin
        function init_cacheval(
                ::$(alg), ::Symmetric{<:Number, <:BandedMatrix}, b, u, Pl,
                Pr, maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
                assumptions::OperatorAssumptions
            )
            return nothing
        end
    end
end

end
