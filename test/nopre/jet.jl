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
    # Working tests - these pass JET optimization checks
    JET.@test_opt init(prob, nothing)
    JET.@test_opt solve(prob, LUFactorization())
    JET.@test_opt solve(prob, GenericLUFactorization())
    JET.@test_opt solve(prob, DiagonalFactorization())
    JET.@test_opt solve(prob, SimpleLUFactorization())
    JET.@test_opt solve(prob_spd, NormalCholeskyFactorization())
    JET.@test_opt solve(prob, NormalBunchKaufmanFactorization())
    
    # The following tests are currently commented out due to type stability issues:
    #
    # QRFactorization - runtime dispatch in LinearAlgebra.QRCompactWYQ
    # Issue: getproperty(F::QRCompactWY, d::Symbol) has runtime dispatch
    # JET.@test_opt solve(prob, QRFactorization())
    #
    # CholeskyFactorization - type instability
    # JET.@test_opt solve(prob_spd, CholeskyFactorization())
    #
    # LDLtFactorization - type instability  
    # JET.@test_opt solve(prob_sym, LDLtFactorization())
    #
    # SVDFactorization - may pass on some Julia versions
    # JET.@test_opt solve(prob, SVDFactorization())
    #
    # BunchKaufmanFactorization - type instability
    # JET.@test_opt solve(prob_sym, BunchKaufmanFactorization())
    #
    # GenericFactorization - runtime dispatch in issuccess
    # Issue: _notsuccessful(F::LU) and hasmethod checks cause runtime dispatch
    # JET.@test_opt solve(prob, GenericFactorization())
end

@testset "JET Tests for Extension Factorizations" begin
    # RecursiveFactorization.jl extensions
    JET.@test_opt solve(prob, RFLUFactorization())
    
    # The following tests are currently commented out due to type stability issues:
    #
    # FastLUFactorization - runtime dispatch in do_factorization
    # JET.@test_opt solve(prob, FastLUFactorization())
    #
    # FastQRFactorization - type instability
    # JET.@test_opt solve(prob, FastQRFactorization())
    #
    # Platform-specific factorizations would go here if enabled:
    # MKLLUFactorization, AppleAccelerateLUFactorization, etc.
end

@testset "JET Tests for Sparse Factorizations" begin
    # All sparse factorizations currently have type stability issues
    # These tests are disabled until the issues are resolved:
    #
    # UMFPACKFactorization - type instability
    # JET.@test_opt solve(prob_sparse, UMFPACKFactorization())
    #
    # KLUFactorization - type instability  
    # JET.@test_opt solve(prob_sparse, KLUFactorization())
    #
    # CHOLMODFactorization - type instability
    # JET.@test_opt solve(prob_sparse_spd, CHOLMODFactorization())
    #
    # SparspakFactorization - type instability
    # JET.@test_opt solve(prob_sparse, SparspakFactorization())
end

@testset "JET Tests for Krylov Methods" begin
    # All Krylov methods currently have type stability issues
    # These tests are disabled until the issues are resolved:
    #
    # KrylovJL_GMRES - type instability
    # JET.@test_opt solve(prob, KrylovJL_GMRES())
    #
    # KrylovJL_CG - type instability
    # JET.@test_opt solve(prob_spd, KrylovJL_CG())
    #
    # KrylovJL_MINRES - type instability
    # JET.@test_opt solve(prob_sym, KrylovJL_MINRES())
    #
    # KrylovJL_BICGSTAB - type instability
    # JET.@test_opt solve(prob, KrylovJL_BICGSTAB())
    #
    # KrylovJL_LSMR - type instability
    # JET.@test_opt solve(prob, KrylovJL_LSMR())
    #
    # KrylovJL_CRAIGMR - type instability
    # JET.@test_opt solve(prob, KrylovJL_CRAIGMR())
    #
    # KrylovJL_MINARES - type instability
    # JET.@test_opt solve(prob_sym, KrylovJL_MINARES())
    #
    # SimpleGMRES - type instability
    # JET.@test_opt solve(prob, SimpleGMRES())
end

@testset "JET Tests for Default Solver" begin
    # Default solver selection has runtime dispatch issues
    # These tests are disabled until the issues are resolved:
    #
    # Default solver - runtime dispatch in solver selection
    # JET.@test_opt solve(prob)
    # JET.@test_opt solve(prob_sparse)
end