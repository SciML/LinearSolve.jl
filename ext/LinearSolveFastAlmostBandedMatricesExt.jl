module LinearSolveFastAlmostBandedMatricesExt

using FastAlmostBandedMatrices, LinearAlgebra, LinearSolve
import LinearSolve: defaultalg,
                    do_factorization, init_cacheval, DefaultLinearSolver,
                    DefaultAlgorithmChoice

function defaultalg(A::AlmostBandedMatrix, b, oa::OperatorAssumptions{Bool})
    if oa.issq
        return DefaultLinearSolver(DefaultAlgorithmChoice.DirectLdiv!)
    else
        return DefaultLinearSolver(DefaultAlgorithmChoice.QRFactorization)
    end
end

# For BandedMatrix
for alg in (:SVDFactorization, :MKLLUFactorization, :DiagonalFactorization,
    :SparspakFactorization, :KLUFactorization, :UMFPACKFactorization,
    :GenericLUFactorization, :RFLUFactorization, :BunchKaufmanFactorization,
    :CHOLMODFactorization, :NormalCholeskyFactorization, :LDLtFactorization,
    :AppleAccelerateLUFactorization, :CholeskyFactorization, :LUFactorization)
    @eval begin
        function init_cacheval(::$(alg), ::AlmostBandedMatrix, b, u, Pl, Pr, maxiters::Int,
                abstol, reltol, verbose::Bool, assumptions::OperatorAssumptions)
            return nothing
        end
    end
end

function do_factorization(alg::QRFactorization, A::AlmostBandedMatrix, b, u)
    return alg.inplace ? qr!(A) : qr(A)
end

end
