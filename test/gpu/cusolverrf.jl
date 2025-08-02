using LinearSolve
using CUSOLVERRF
using CUDA
using SparseArrays
using LinearAlgebra
using Test

@testset "CUSOLVERRFFactorization" begin
    # Skip tests if CUDA is not available
    if !CUDA.functional()
        @info "CUDA not available, skipping CUSOLVERRF tests"
        return
    end
    
    # Test with a small sparse matrix
    n = 100
    A = sprand(n, n, 0.1) + I
    b = rand(n)
    
    # Test with CPU sparse matrix (should auto-convert to GPU)
    @testset "CPU Sparse Matrix" begin
        prob = LinearProblem(A, b)
        
        # Test with default symbolic (:RF)
        sol = solve(prob, CUSOLVERRFFactorization())
        @test norm(A * sol.u - b) / norm(b) < 1e-10
        
        # Test with KLU symbolic
        sol_klu = solve(prob, CUSOLVERRFFactorization(symbolic = :KLU))
        @test norm(A * sol_klu.u - b) / norm(b) < 1e-10
    end
    
    # Test with GPU sparse matrix
    @testset "GPU Sparse Matrix" begin
        A_gpu = CUDA.CUSPARSE.CuSparseMatrixCSR(A)
        b_gpu = CuArray(b)
        
        prob_gpu = LinearProblem(A_gpu, b_gpu)
        sol_gpu = solve(prob_gpu, CUSOLVERRFFactorization())
        
        # Check residual on GPU
        res_gpu = A_gpu * sol_gpu.u - b_gpu
        @test norm(res_gpu) / norm(b_gpu) < 1e-10
    end
    
    # Test matrix update with same sparsity pattern
    @testset "Matrix Update" begin
        # Create a new matrix with same pattern but different values
        A2 = A + 0.1 * sprand(n, n, 0.01)
        b2 = rand(n)
        
        prob2 = LinearProblem(A2, b2)
        sol2 = solve(prob2, CUSOLVERRFFactorization(reuse_symbolic = true))
        @test norm(A2 * sol2.u - b2) / norm(b2) < 1e-10
    end
    
    # Test multiple right-hand sides
    @testset "Multiple RHS" begin
        nrhs = 5
        B = rand(n, nrhs)
        
        prob_multi = LinearProblem(A, B)
        sol_multi = solve(prob_multi, CUSOLVERRFFactorization())
        
        # Check each solution
        for i in 1:nrhs
            @test norm(A * sol_multi.u[:, i] - B[:, i]) / norm(B[:, i]) < 1e-10
        end
    end
    
    # Test adjoint solve
    @testset "Adjoint Solve" begin
        prob_adj = LinearProblem(A', b)
        sol_adj = solve(prob_adj, CUSOLVERRFFactorization())
        @test norm(A' * sol_adj.u - b) / norm(b) < 1e-10
    end
    
    # Test error handling for unsupported types
    @testset "Error Handling" begin
        # Test with Float32 (not supported)
        A_f32 = Float32.(A)
        b_f32 = Float32.(b)
        prob_f32 = LinearProblem(A_f32, b_f32)
        
        # This should error since CUSOLVERRF only supports Float64
        @test_throws Exception solve(prob_f32, CUSOLVERRFFactorization())
    end
end