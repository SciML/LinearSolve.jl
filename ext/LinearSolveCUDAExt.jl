module LinearSolveCUDAExt

using CUDA, LinearAlgebra, LinearSolve, SciMLBase
using SciMLBase: AbstractSciMLOperator

function SciMLBase.solve!(cache::LinearSolve.LinearCache, alg::CudaOffloadFactorization;
    kwargs...)
    if cache.isfresh
        fact = qr(CUDA.CuArray(cache.A))
        cache.cacheval = fact
        cache.isfresh = false
    end
    y = Array(ldiv!(cache.u, cache.cacheval, CUDA.CuArray(cache.u)))
    SciMLBase.build_linear_solution(alg, y, nothing, cache)
end

function LinearSolve.init_cacheval(alg::CudaOffloadFactorization, A, b, u, Pl, Pr,
    maxiters::Int, abstol, reltol, verbose::Bool,
    assumptions::OperatorAssumptions)
    ArrayInterface.lu_instance(CUDA.CuArray(A))
end

end
