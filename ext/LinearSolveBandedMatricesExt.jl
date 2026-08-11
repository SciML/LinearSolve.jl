module LinearSolveBandedMatricesExt

using ArrayInterface: ArrayInterface
using BandedMatrices: BandedMatrices, BandedMatrix
using LinearAlgebra: LinearAlgebra, ColumnNorm, NoPivot, Symmetric, lu, lu!, qr, qr!
# The `@eval` loops below generate `init_cacheval` methods for every algorithm type,
# which ExplicitImports cannot see, so they are all imported explicitly here.
using LinearSolve: LinearSolve, LUFactorization, OperatorAssumptions, QRFactorization,
    AppleAccelerateLUFactorization, BunchKaufmanFactorization, CHOLMODFactorization,
    CholeskyFactorization, DiagonalFactorization, GenericLUFactorization,
    KLUFactorization, LDLtFactorization, MKLLUFactorization,
    NormalCholeskyFactorization, RFLUFactorization, SVDFactorization,
    SparspakFactorization, UMFPACKFactorization
import LinearSolve: defaultalg,
    do_factorization, init_cacheval, DefaultLinearSolver,
    DefaultAlgorithmChoice, LinearVerbosity

# Defaults for BandedMatrices
function defaultalg(A::BandedMatrix, b, oa::OperatorAssumptions{Bool})
    if oa.issq
        return DefaultLinearSolver(DefaultAlgorithmChoice.LUFactorization)
    end
    # Both non-square cases go to QR. The underdetermined one is solved through
    # the QR of `Aᵀ`; see `BandedUnderdeterminedQR`.
    return DefaultLinearSolver(DefaultAlgorithmChoice.QRFactorization)
end

function defaultalg(
        A::BandedMatrix{T}, b, oa::OperatorAssumptions{Bool}
    ) where {T <: BigFloat}
    return DefaultLinearSolver(DefaultAlgorithmChoice.QRFactorization)
end

function defaultalg(A::Symmetric{<:Number, <:BandedMatrix}, b, ::OperatorAssumptions{Bool})
    return DefaultLinearSolver(DefaultAlgorithmChoice.CholeskyFactorization)
end

raw"""
    BandedUnderdeterminedQR(qr_of_transpose)

The QR factorization of `transpose(A)` for an underdetermined (wide)
`A::BandedMatrix`.

BandedMatrices can factor a wide banded matrix but cannot solve with the result:
`A \ b` throws `"Not implemented"`. The minimum-norm solution comes from the QR
of `Aᵀ`, which is banded as well (the bandwidths simply swap), so the banded
factorization still does the work rather than falling back to a dense one:

```
Aᵀ = Q R   ⟹   A = Rᵀ Qᵀ   ⟹   x = Q [Rᵀ \ b; 0]
```

That is the same minimum-norm solution LAPACK's dense `\` returns for an
underdetermined system. See SciML/LinearSolve.jl#419.
"""
struct BandedUnderdeterminedQR{F}
    qr_of_transpose::F
end

function LinearSolve._ldiv!(x, F::BandedUnderdeterminedQR, b)
    nrows = length(b)
    # `R` is upper triangular and square in its leading `nrows` block; the rows
    # below it are structurally zero and contribute nothing to the solve.
    R = view(F.qr_of_transpose.R, 1:nrows, 1:nrows)
    y = LinearAlgebra.ldiv!(
        LinearAlgebra.adjoint(LinearAlgebra.UpperTriangular(R)),
        copyto!(similar(x, nrows), b)
    )
    # Pad with the zeros that make `x` the minimum-norm solution, then apply `Q`.
    fill!(x, zero(eltype(x)))
    copyto!(view(x, 1:nrows), y)
    LinearAlgebra.lmul!(F.qr_of_transpose.Q, x)
    return x
end

# BandedMatrices `qr` doesn't support column pivoting, so convert to dense when
# pivoting is requested (e.g. ColumnNorm fallback from singular LU).
function do_factorization(alg::QRFactorization, A::BandedMatrix, b, u)
    if !(alg.pivot isa NoPivot)
        return qr!(Matrix(A), alg.pivot)
    elseif LinearSolve.is_underdetermined(A)
        # `qr!` would factor `A` itself, which cannot then be solved with, so the
        # transpose is always built fresh here regardless of `alg.inplace`.
        return BandedUnderdeterminedQR(qr(BandedMatrix(transpose(A))))
    else
        return alg.inplace ? qr!(A) : qr(A)
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

# `cache.cacheval` is typed from this, and the underdetermined path stores a
# `BandedUnderdeterminedQR` rather than a plain `QR`, so the placeholder has to be
# that same wrapper. Square and overdetermined `A` keep the generic instance.
function init_cacheval(
        alg::QRFactorization{NoPivot}, A::BandedMatrix, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    )
    LinearSolve.is_underdetermined(A) ||
        return ArrayInterface.qr_instance(A, alg.pivot)
    return BandedUnderdeterminedQR(qr(BandedMatrix(transpose(similar(A, 0, 0)))))
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
