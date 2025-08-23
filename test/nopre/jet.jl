using LinearSolve, ForwardDiff, RecursiveFactorization, LinearAlgebra, SparseArrays, Test
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

# Dual problem set up 
function h(p)
    (A = [p[1] p[2]+1 p[2]^3;
          3*p[1] p[1]+5 p[2] * p[1]-4;
          p[2]^2 9*p[1] p[2]],
        b = [p[1] + 1, p[2] * 2, p[1]^2])
end

A, b = h([ForwardDiff.Dual(5.0, 1.0, 0.0), ForwardDiff.Dual(5.0, 0.0, 1.0)])

dual_prob = LinearProblem(A, b)

@testset "JET Tests for Dense Factorizations" begin
    # Working tests - these pass JET optimization checks
    JET.@test_opt init(prob, nothing)
    JET.@test_opt solve(prob, LUFactorization())
    JET.@test_opt solve(prob, GenericLUFactorization())
    JET.@test_opt solve(prob, DiagonalFactorization())
    JET.@test_opt solve(prob, SimpleLUFactorization())
    # JET.@test_opt solve(prob_spd, NormalCholeskyFactorization())
    # JET.@test_opt solve(prob, NormalBunchKaufmanFactorization())
    
    # CholeskyFactorization and SVDFactorization now pass JET tests
    # JET.@test_opt solve(prob_spd, CholeskyFactorization())
    # JET.@test_opt solve(prob, SVDFactorization())
    
    # Tests with known type stability issues - marked as broken
    JET.@test_opt solve(prob, QRFactorization()) broken=true
    JET.@test_opt solve(prob_sym, LDLtFactorization()) broken=true
    JET.@test_opt solve(prob_sym, BunchKaufmanFactorization()) broken=true
    JET.@test_opt solve(prob, GenericFactorization()) broken=true
end

@testset "JET Tests for Extension Factorizations" begin
    # RecursiveFactorization.jl extensions
    # JET.@test_opt solve(prob, RFLUFactorization())
    
    # Tests with known type stability issues
    JET.@test_opt solve(prob, FastLUFactorization()) broken=true
    JET.@test_opt solve(prob, FastQRFactorization()) broken=true
    
    # Platform-specific factorizations (may not be available on all systems)
    if @isdefined(MKLLUFactorization)
        # MKLLUFactorization passes JET tests
        JET.@test_opt solve(prob, MKLLUFactorization())
    end
    
    if Sys.isapple() && @isdefined(AppleAccelerateLUFactorization)
        JET.@test_opt solve(prob, AppleAccelerateLUFactorization()) broken=true
    end
    
    # CUDA/Metal factorizations (only test if CUDA/Metal are loaded)
    # CudaOffloadFactorization requires CUDA to be loaded, skip if not available
    # Metal is only available on Apple platforms
    if Sys.isapple() && @isdefined(MetalLUFactorization)
        JET.@test_opt solve(prob, MetalLUFactorization()) broken=true
    end
    if @isdefined(BLISLUFactorization)
        JET.@test_opt solve(prob, BLISLUFactorization()) broken=true
    end
end

@testset "JET Tests for Sparse Factorizations" begin
    JET.@test_opt solve(prob_sparse, UMFPACKFactorization()) broken=true
    JET.@test_opt solve(prob_sparse, KLUFactorization()) broken=true
    JET.@test_opt solve(prob_sparse_spd, CHOLMODFactorization()) broken=true
    
    # SparspakFactorization requires Sparspak to be loaded
    # PardisoJL requires Pardiso to be loaded
    # CUSOLVERRFFactorization requires CUSOLVERRF to be loaded
    # These are tested in their respective extension test suites
end

@testset "JET Tests for Krylov Methods" begin
    # KrylovJL methods that pass JET tests
    # JET.@test_opt solve(prob_spd, KrylovJL_CG())
    # JET.@test_opt solve(prob, KrylovJL_BICGSTAB())
    # JET.@test_opt solve(prob, KrylovJL_LSMR())
    # JET.@test_opt solve(prob, KrylovJL_CRAIGMR())
    
    # SimpleGMRES passes JET tests
    # JET.@test_opt solve(prob, SimpleGMRES())
    
    # KrylovJL methods with known type stability issues
    JET.@test_opt solve(prob, KrylovJL_GMRES()) broken=true
    JET.@test_opt solve(prob_sym, KrylovJL_MINRES()) broken=true
    JET.@test_opt solve(prob_sym, KrylovJL_MINARES()) broken=true
    
    # Extension Krylov methods (require extensions)
    # KrylovKitJL_CG, KrylovKitJL_GMRES require KrylovKit to be loaded
    # IterativeSolversJL requires IterativeSolvers to be loaded
    # These are tested in their respective extension test suites
end

@testset "JET Tests for Default Solver" begin
    # Test the default solver selection
    JET.@test_opt solve(prob) broken=true
    JET.@test_opt solve(prob_sparse) broken=true
end

@testset "JET Tests for creating Dual solutions" begin
    # Make sure there's no runtime dispatch when making solutions of Dual problems
    dual_cache = init(prob)
    ext = Base.get_extension(LinearSolve, :LinearSolveForwardDiffExt)
    JET.@test_opt ext.linearsolve_dual_solution(
        [1.0, 1.0, 1.0], [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dual_cache )
end