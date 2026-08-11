module LinearSolveFastAlmostBandedMatricesExt

using ArrayInterface: ArrayInterface
using FastAlmostBandedMatrices: FastAlmostBandedMatrices, AlmostBandedMatrix
using LinearAlgebra: LinearAlgebra, NoPivot, qr, qr!
# The `@eval` loop below generates `init_cacheval` methods for every algorithm type,
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

function defaultalg(A::AlmostBandedMatrix, b, oa::OperatorAssumptions{Bool})
    if oa.issq
        return DefaultLinearSolver(DefaultAlgorithmChoice.DirectLdiv!)
    else
        return DefaultLinearSolver(DefaultAlgorithmChoice.QRFactorization)
    end
end

# `cache.cacheval` is typed from this, and the underdetermined path stores a
# `MinNormQR` rather than a plain `QR`, so the placeholder has to be that same
# wrapper. Square and overdetermined `A` keep the generic instance.
function init_cacheval(
        alg::QRFactorization{NoPivot}, A::AlmostBandedMatrix, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    )
    LinearSolve.is_underdetermined(A) ||
        return ArrayInterface.qr_instance(A, alg.pivot)
    return LinearSolve.MinNormQR(qr(transpose(similar(A, 0, 0))))
end

# For BandedMatrix
for alg in (
        :SVDFactorization, :MKLLUFactorization, :DiagonalFactorization,
        :SparspakFactorization, :KLUFactorization, :UMFPACKFactorization,
        :GenericLUFactorization, :RFLUFactorization, :BunchKaufmanFactorization,
        :CHOLMODFactorization, :NormalCholeskyFactorization, :LDLtFactorization,
        :AppleAccelerateLUFactorization, :CholeskyFactorization, :LUFactorization,
    )
    @eval begin
        function init_cacheval(
                ::$(alg), ::AlmostBandedMatrix, b, u, Pl, Pr, maxiters::Int,
                abstol, reltol, verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
            )
            return nothing
        end
    end
end

function do_factorization(alg::QRFactorization, A::AlmostBandedMatrix, b, u)
    if LinearSolve.is_underdetermined(A)
        # The same "Not implemented" wall as `BandedMatrix`, solved the same way,
        # except the transpose cannot stay structured: an `AlmostBandedMatrix` is
        # banded plus dense fill *rows*, so its transpose has dense fill *columns*
        # and is not almost-banded. `qr` of the lazy transpose densifies, which is
        # the price of returning the same minimum-norm solution the dense and
        # banded paths give. (`BandedMatrix(transpose(A))` would stay structured
        # but silently discards the fill rows, giving the wrong matrix.)
        return LinearSolve.MinNormQR(qr(transpose(A)))
    end
    return alg.inplace ? qr!(A) : qr(A)
end

end
