module LinearSolve

using SciMLBase: AbstractLinearAlgorithm, AbstractDiffEqOperator
using Reexport
using LinearAlgebra
@reexport using SciMLBase

export LUFactorization, QRFactorization, SVDFactorization

struct LUFactorization{P} <: AbstractLinearAlgorithm
    pivot::P
end

function SciMLBase.solve(prob::LinearProblem, alg::LUFactorization)
    prob.A isa Union{AbstractMatrix, AbstractDiffEqOperator} || error("LU is not defined for $(typeof(prob.A))")
    lu(prob.A, alg.pivot) \ prob.b
end

struct QRFactorization{P} <: AbstractLinearAlgorithm
    pivot::P
    blocksize::Int
end

function SciMLBase.solve(prob::LinearProblem, alg::QRFactorization)
    prob.A isa Union{AbstractMatrix, AbstractDiffEqOperator} || error("QR is not defined for $(typeof(prob.A))")
    qr(prob.A, alg.pivot; blocksize=alg.blocksize) \ prob.b
end

struct SVDFactorization{A} <: AbstractLinearAlgorithm
    full::Bool
    alg::A
end

function SciMLBase.solve(prob::LinearProblem, ::SVDFactorization)
    prob.A isa Union{AbstractMatrix, AbstractDiffEqOperator} || error("SVD is not defined for $(typeof(prob.A))")
    svd(prob.A; full=alg.full, alg=alg.alg) \ prob.b
end

end
