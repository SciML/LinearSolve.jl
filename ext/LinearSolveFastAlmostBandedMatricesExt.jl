module LinearSolveFastAlmostBandedMatricesExt

using FastAlmostBandedMatrices: FastAlmostBandedMatrices, AlmostBandedMatrix
using LinearAlgebra: LinearAlgebra, qr, qr!
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
    return alg.inplace ? qr!(A) : qr(A)
end

end
