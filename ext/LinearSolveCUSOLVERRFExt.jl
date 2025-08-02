module LinearSolveCUSOLVERRFExt

using LinearSolve: LinearSolve, @get_cacheval, pattern_changed, OperatorAssumptions
using CUSOLVERRF: CUSOLVERRF, RFLU
using SparseArrays: SparseArrays, SparseMatrixCSC, nnz
using CUDA: CUDA
using CUDA.CUSPARSE: CuSparseMatrixCSR
using LinearAlgebra: LinearAlgebra, ldiv!, lu!
using SciMLBase: SciMLBase, LinearProblem, ReturnCode

function LinearSolve.init_cacheval(alg::LinearSolve.CUSOLVERRFFactorization,
        A, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Bool, assumptions::OperatorAssumptions)
    nothing
end

function LinearSolve.init_cacheval(alg::LinearSolve.CUSOLVERRFFactorization,
        A::Union{CuSparseMatrixCSR{Float64, Int32}, SparseMatrixCSC{Float64, <:Integer}}, 
        b, u, Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Bool, assumptions::OperatorAssumptions)
    # Create initial factorization with appropriate options
    nrhs = b isa AbstractMatrix ? size(b, 2) : 1
    symbolic = alg.symbolic
    # Convert to CuSparseMatrixCSR if needed
    A_gpu = A isa CuSparseMatrixCSR ? A : CuSparseMatrixCSR(A)
    RFLU(A_gpu; nrhs=nrhs, symbolic=symbolic)
end

function SciMLBase.solve!(cache::LinearSolve.LinearCache, alg::LinearSolve.CUSOLVERRFFactorization; kwargs...)
    A = cache.A
    
    # Convert to appropriate GPU format if needed
    if A isa SparseMatrixCSC
        A_gpu = CuSparseMatrixCSR(A)
    elseif A isa CuSparseMatrixCSR
        A_gpu = A
    else
        error("CUSOLVERRFFactorization only supports SparseMatrixCSC or CuSparseMatrixCSR matrices")
    end
    
    if cache.isfresh
        cacheval = @get_cacheval(cache, :CUSOLVERRFFactorization)
        if cacheval === nothing
            # Create new factorization
            nrhs = cache.b isa AbstractMatrix ? size(cache.b, 2) : 1
            fact = RFLU(A_gpu; nrhs=nrhs, symbolic=alg.symbolic)
        else
            # Reuse symbolic factorization if pattern hasn't changed
            if alg.reuse_symbolic && !pattern_changed(cacheval, A_gpu)
                fact = cacheval
                lu!(fact, A_gpu)
            else
                # Create new factorization if pattern changed
                nrhs = cache.b isa AbstractMatrix ? size(cache.b, 2) : 1
                fact = RFLU(A_gpu; nrhs=nrhs, symbolic=alg.symbolic)
            end
        end
        cache.cacheval = fact
        cache.isfresh = false
    end
    
    F = @get_cacheval(cache, :CUSOLVERRFFactorization)
    
    # Ensure b and u are on GPU
    b_gpu = cache.b isa CUDA.CuArray ? cache.b : CUDA.CuArray(cache.b)
    u_gpu = cache.u isa CUDA.CuArray ? cache.u : CUDA.CuArray(cache.u)
    
    # Solve
    ldiv!(u_gpu, F, b_gpu)
    
    # Copy back to CPU if needed
    if !(cache.u isa CUDA.CuArray)
        copyto!(cache.u, u_gpu)
    end
    
    SciMLBase.build_linear_solution(alg, cache.u, nothing, cache; retcode = ReturnCode.Success)
end

# Helper function for pattern checking
function LinearSolve.pattern_changed(rf::RFLU, A::CuSparseMatrixCSR)
    # For CUSOLVERRF, we need to check if the sparsity pattern has changed
    # This is a simplified check - you might need a more sophisticated approach
    size(rf) != size(A) || nnz(rf.M) != nnz(A)
end

# Extension load check
LinearSolve.cusolverrf_loaded(A::CuSparseMatrixCSR) = true
LinearSolve.cusolverrf_loaded(A::SparseMatrixCSC{Float64}) = true

end