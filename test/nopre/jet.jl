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
    JET.@test_opt solve(prob_spd, NormalCholeskyFactorization())
    JET.@test_opt solve(prob, NormalBunchKaufmanFactorization())
    
    # TODO: Fix type stability issues in these solvers:
    # - QRFactorization: runtime dispatch in LinearAlgebra.QRCompactWYQ
    # - CholeskyFactorization: type instability issues
    # - LDLtFactorization: type instability issues
    # - SVDFactorization: type instability issues
    # - BunchKaufmanFactorization: type instability issues
    # - GenericFactorization: runtime dispatch in issuccess
    
    # Uncomment these once type stability is fixed:
    # JET.@test_opt solve(prob, QRFactorization())
    # JET.@test_opt solve(prob_spd, CholeskyFactorization())
    # JET.@test_opt solve(prob_sym, LDLtFactorization())
    # JET.@test_opt solve(prob, SVDFactorization())
    # JET.@test_opt solve(prob_sym, BunchKaufmanFactorization())
    # JET.@test_opt solve(prob, GenericFactorization())
end

@testset "JET Tests for Extension Factorizations" begin
    # RecursiveFactorization.jl extensions
    JET.@test_opt solve(prob, RFLUFactorization())
    
    # TODO: Fix type stability in FastLUFactorization and FastQRFactorization
    # - FastLUFactorization: runtime dispatch in do_factorization
    # - FastQRFactorization: type instability issues
    
    # Uncomment these once type stability is fixed:
    # JET.@test_opt solve(prob, FastLUFactorization())
    # JET.@test_opt solve(prob, FastQRFactorization())
    
    # Platform-specific factorizations (may not be available on all systems)
    # These need conditional testing based on platform and availability
    # if @isdefined(MKLLUFactorization)
    #     JET.@test_opt solve(prob, MKLLUFactorization())
    # end
    # if Sys.isapple() && @isdefined(AppleAccelerateLUFactorization)
    #     JET.@test_opt solve(prob, AppleAccelerateLUFactorization())
    # end
end

@testset "JET Tests for Sparse Factorizations" begin
    # TODO: Fix type stability issues in sparse factorizations
    # All sparse factorizations currently have type instability issues
    # that need to be addressed before enabling these tests
    
    # Uncomment these once type stability is fixed:
    # JET.@test_opt solve(prob_sparse, UMFPACKFactorization())
    # JET.@test_opt solve(prob_sparse, KLUFactorization())
    # JET.@test_opt solve(prob_sparse_spd, CHOLMODFactorization())
    # JET.@test_opt solve(prob_sparse, SparspakFactorization())
end

@testset "JET Tests for Krylov Methods" begin
    # TODO: Fix type stability issues in Krylov methods
    # All Krylov methods currently have type instability issues
    # that need to be addressed before enabling these tests
    
    # Uncomment these once type stability is fixed:
    # JET.@test_opt solve(prob, KrylovJL_GMRES())
    # JET.@test_opt solve(prob_spd, KrylovJL_CG())
    # JET.@test_opt solve(prob_sym, KrylovJL_MINRES())
    # JET.@test_opt solve(prob, KrylovJL_BICGSTAB())
    # JET.@test_opt solve(prob, KrylovJL_LSMR())
    # JET.@test_opt solve(prob, KrylovJL_CRAIGMR())
    # JET.@test_opt solve(prob_sym, KrylovJL_MINARES())
    # JET.@test_opt solve(prob, SimpleGMRES())
end

@testset "JET Tests for Default Solver" begin
    # TODO: Fix type stability in default solver selection
    # The default solver selection has runtime dispatch issues
    
    # Uncomment these once type stability is fixed:
    # JET.@test_opt solve(prob)
    # JET.@test_opt solve(prob_sparse)
end