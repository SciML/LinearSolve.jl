module LinearSolveCUDAExt

using CUDA
using LinearSolve
using LinearSolve.LinearAlgebra, LinearSolve.SciMLBase, LinearSolve.ArrayInterface
using SciMLBase: AbstractSciMLOperator

function SciMLBase.solve!(cache::LinearSolve.LinearCache, alg::CudaOffloadFactorization;
    kwargs...)
    if cache.isfresh
        fact = qr(CUDA.CuArray(cache.A))
        cache.cacheval = fact
        cache.isfresh = false
    end
    y = Array(ldiv!(CUDA.CuArray(cache.u), cache.cacheval, CUDA.CuArray(cache.b)))
    cache.u .= y
    SciMLBase.build_linear_solution(alg, y, nothing, cache)
end

function LinearSolve.init_cacheval(alg::CudaOffloadFactorization, A, b, u, Pl, Pr,
    maxiters::Int, abstol, reltol, verbose::Bool,
    assumptions::OperatorAssumptions)
    qr(CUDA.CuArray(A))
end

end
