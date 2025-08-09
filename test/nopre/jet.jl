using LinearSolve, RecursiveFactorization, LinearAlgebra, SparseArrays, Test
using JET

# Dense problem setup
A = rand(4, 4)
b = rand(4)
prob = LinearProblem(A, b)

# Symmetric positive definite matrix for Cholesky
A_spd = A' * A + I
prob_spd = LinearProblem(A_spd, b)

# Symmetric matrix for LDLt
A_sym = A + A'
prob_sym = LinearProblem(A_sym, b)

# Sparse problem setup
A_sparse = sparse(A)
prob_sparse = LinearProblem(A_sparse, b)

# Sparse SPD for CHOLMODFactorization
A_sparse_spd = sparse(A_spd)
prob_sparse_spd = LinearProblem(A_sparse_spd, b)

@testset "JET Tests for Dense Factorizations" begin
    # Working tests
    JET.@test_opt init(prob, nothing)
    JET.@test_opt solve(prob, LUFactorization())
    JET.@test_opt solve(prob, GenericLUFactorization())
    JET.@test_opt solve(prob, DiagonalFactorization())
    JET.@test_opt solve(prob, SimpleLUFactorization())
    
    # Tests that currently fail - marked with @test_skip
    @test_skip JET.@test_opt solve(prob, QRFactorization())
    @test_skip JET.@test_opt solve(prob_spd, CholeskyFactorization())
    @test_skip JET.@test_opt solve(prob_sym, LDLtFactorization())
    @test_skip JET.@test_opt solve(prob, SVDFactorization())
    @test_skip JET.@test_opt solve(prob_sym, BunchKaufmanFactorization())
    @test_skip JET.@test_opt solve(prob, GenericFactorization())
    JET.@test_opt solve(prob_spd, NormalCholeskyFactorization())
    JET.@test_opt solve(prob, NormalBunchKaufmanFactorization())
end

@testset "JET Tests for Extension Factorizations" begin
    # RecursiveFactorization.jl extensions
    JET.@test_opt solve(prob, RFLUFactorization())
    @test_skip JET.@test_opt solve(prob, FastLUFactorization())
    @test_skip JET.@test_opt solve(prob, FastQRFactorization())
    
    # Platform-specific factorizations (may not be available on all systems)
    if @isdefined(MKLLUFactorization)
        @test_skip JET.@test_opt solve(prob, MKLLUFactorization())
    end
    
    if Sys.isapple() && @isdefined(AppleAccelerateLUFactorization)
        @test_skip JET.@test_opt solve(prob, AppleAccelerateLUFactorization())
    end
    
    # CUDA/Metal factorizations (only test if available)
    # @test_skip JET.@test_opt solve(prob, CudaOffloadFactorization())
    # @test_skip JET.@test_opt solve(prob, MetalLUFactorization())
    # @test_skip JET.@test_opt solve(prob, BLISLUFactorization())
end

@testset "JET Tests for Sparse Factorizations" begin
    @test_skip JET.@test_opt solve(prob_sparse, UMFPACKFactorization())
    @test_skip JET.@test_opt solve(prob_sparse, KLUFactorization())
    @test_skip JET.@test_opt solve(prob_sparse_spd, CHOLMODFactorization())
    @test_skip JET.@test_opt solve(prob_sparse, SparspakFactorization())
    
    # PardisoJL (requires extension)
    # @test_skip JET.@test_opt solve(prob_sparse, PardisoJL())
    
    # CUSOLVER (requires CUDA)
    # @test_skip JET.@test_opt solve(prob_sparse, CUSOLVERRFFactorization())
end

@testset "JET Tests for Krylov Methods" begin
    # KrylovJL methods
    @test_skip JET.@test_opt solve(prob, KrylovJL_GMRES())
    @test_skip JET.@test_opt solve(prob_spd, KrylovJL_CG())
    @test_skip JET.@test_opt solve(prob_sym, KrylovJL_MINRES())
    @test_skip JET.@test_opt solve(prob, KrylovJL_BICGSTAB())
    @test_skip JET.@test_opt solve(prob, KrylovJL_LSMR())
    @test_skip JET.@test_opt solve(prob, KrylovJL_CRAIGMR())
    @test_skip JET.@test_opt solve(prob_sym, KrylovJL_MINARES())
    
    # SimpleGMRES
    @test_skip JET.@test_opt solve(prob, SimpleGMRES())
    
    # Extension Krylov methods (require extensions)
    # @test_skip JET.@test_opt solve(prob, KrylovKitJL_CG())
    # @test_skip JET.@test_opt solve(prob, KrylovKitJL_GMRES())
    # @test_skip JET.@test_opt solve(prob, IterativeSolversJL())
end

@testset "JET Tests for Default Solver" begin
    # Test the default solver selection
    @test_skip JET.@test_opt solve(prob)
    @test_skip JET.@test_opt solve(prob_sparse)
end