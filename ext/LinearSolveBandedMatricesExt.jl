module LinearSolveBandedMatricesExt

using BandedMatrices, LinearAlgebra, LinearSolve
import LinearSolve: defaultalg,
                    do_factorization, init_cacheval, DefaultLinearSolver,
                    DefaultAlgorithmChoice

# Defaults for BandedMatrices
function defaultalg(A::BandedMatrix, b, oa::OperatorAssumptions{Bool})
    if oa.issq
        return DefaultLinearSolver(DefaultAlgorithmChoice.DirectLdiv!)
    elseif LinearSolve.is_underdetermined(A)
        error("No solver for underdetermined `A::BandedMatrix` is currently implemented!")
    else
        return DefaultLinearSolver(DefaultAlgorithmChoice.QRFactorization)
    end
end

function defaultalg(A::Symmetric{<:Number, <:BandedMatrix}, b, ::OperatorAssumptions{Bool})
    return DefaultLinearSolver(DefaultAlgorithmChoice.CholeskyFactorization)
end

# BandedMatrices `qr` doesn't allow other args without causing an ambiguity
do_factorization(alg::QRFactorization, A::BandedMatrix, b, u) = alg.inplace ? qr!(A) : qr(A)

function do_factorization(alg::LUFactorization, A::BandedMatrix, b, u)
    _pivot = alg.pivot isa NoPivot ? Val(false) : Val(true)
    return lu!(A, _pivot; check = false)
end

# For BandedMatrix
for alg in (:SVDFactorization, :MKLLUFactorization, :DiagonalFactorization,
    :SparspakFactorization, :KLUFactorization, :UMFPACKFactorization,
    :GenericLUFactorization, :RFLUFactorization, :BunchKaufmanFactorization,
    :CHOLMODFactorization, :NormalCholeskyFactorization, :LDLtFactorization,
    :AppleAccelerateLUFactorization, :CholeskyFactorization)
    @eval begin
        function init_cacheval(::$(alg), ::BandedMatrix, b, u, Pl, Pr, maxiters::Int,
                abstol, reltol, verbose::Bool, assumptions::OperatorAssumptions)
            return nothing
        end
    end
end

function init_cacheval(::LUFactorization, A::BandedMatrix, b, u, Pl, Pr, maxiters::Int,
        abstol, reltol, verbose::Bool, assumptions::OperatorAssumptions)
    return lu(similar(A, 0, 0))
end

# For Symmetric BandedMatrix
for alg in (:SVDFactorization, :MKLLUFactorization, :DiagonalFactorization,
    :SparspakFactorization, :KLUFactorization, :UMFPACKFactorization,
    :GenericLUFactorization, :RFLUFactorization, :BunchKaufmanFactorization,
    :CHOLMODFactorization, :NormalCholeskyFactorization,
    :AppleAccelerateLUFactorization, :QRFactorization, :LUFactorization)
    @eval begin
        function init_cacheval(::$(alg), ::Symmetric{<:Number, <:BandedMatrix}, b, u, Pl,
                Pr, maxiters::Int, abstol, reltol, verbose::Bool,
                assumptions::OperatorAssumptions)
            return nothing
        end
    end
end

end
