module LinearSolveAMDGPUExt

using AMDGPU
using LinearSolve: LinearSolve, LinearCache, AMDGPUOffloadLUFactorization,
                   AMDGPUOffloadQRFactorization, init_cacheval, OperatorAssumptions
using LinearSolve.LinearAlgebra, LinearSolve.SciMLBase

# LU Factorization
function SciMLBase.solve!(cache::LinearSolve.LinearCache, alg::AMDGPUOffloadLUFactorization;
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

function LinearSolve.init_cacheval(alg::AMDGPUOffloadLUFactorization, A, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::LinearVerbosity,
        assumptions::OperatorAssumptions)
    AMDGPU.rocSOLVER.getrf!(AMDGPU.ROCArray(A))
end

# QR Factorization
function SciMLBase.solve!(cache::LinearSolve.LinearCache, alg::AMDGPUOffloadQRFactorization;
        kwargs...)
    if cache.isfresh
        A_gpu = AMDGPU.ROCArray(cache.A)
        tau = AMDGPU.ROCVector{eltype(A_gpu)}(undef, min(size(A_gpu)...))
        AMDGPU.rocSOLVER.geqrf!(A_gpu, tau)
        cache.cacheval = (A_gpu, tau)
        cache.isfresh = false
    end
    
    A_gpu, tau = cache.cacheval
    b_gpu = AMDGPU.ROCArray(cache.b)
    
    # Apply Q^T to b
    AMDGPU.rocSOLVER.ormqr!('L', 'T', A_gpu, tau, b_gpu)
    
    # Solve the upper triangular system
    m, n = size(A_gpu)
    AMDGPU.rocBLAS.trsv!('U', 'N', 'N', n, A_gpu, b_gpu)
    
    y = Array(b_gpu[1:n])
    cache.u .= y
    SciMLBase.build_linear_solution(alg, y, nothing, cache)
end

function LinearSolve.init_cacheval(alg::AMDGPUOffloadQRFactorization, A, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::LinearVerbosity,
        assumptions::OperatorAssumptions)
    A_gpu = AMDGPU.ROCArray(A)
    tau = AMDGPU.ROCVector{eltype(A_gpu)}(undef, min(size(A_gpu)...))
    AMDGPU.rocSOLVER.geqrf!(A_gpu, tau)
    (A_gpu, tau)
end

end
