module LinearSolveAMDGPUExt

using AMDGPU
using LinearSolve: LinearSolve, LinearCache, AMDGPUOffloadFactorization,
                   init_cacheval, OperatorAssumptions
using LinearSolve.LinearAlgebra, LinearSolve.SciMLBase

function SciMLBase.solve!(cache::LinearSolve.LinearCache, alg::AMDGPUOffloadFactorization;
        kwargs...)
    if cache.isfresh
        fact = AMDGPU.rocSOLVER.getrf!(AMDGPU.ROCArray(cache.A))
        cache.cacheval = fact
        cache.isfresh = false
    end
    
    A_gpu, ipiv = cache.cacheval
    b_gpu = AMDGPU.ROCArray(cache.b)
    
    AMDGPU.rocSOLVER.getrs!('N', A_gpu, ipiv, b_gpu)
    
    y = Array(b_gpu)
    cache.u .= y
    SciMLBase.build_linear_solution(alg, y, nothing, cache)
end

function LinearSolve.init_cacheval(alg::AMDGPUOffloadFactorization, A, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Bool,
        assumptions::OperatorAssumptions)
    AMDGPU.rocSOLVER.getrf!(AMDGPU.ROCArray(A))
end

end