using LinearSolve, ForwardDiff, ForwardDiff, RecursiveFactorization, LinearAlgebra, SparseArrays, Test
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

    # LUFactorization has runtime dispatch in Base.CoreLogging on Julia < 1.11
    # Fixed in Julia 1.11+
    if VERSION < v"1.11"
        JET.@test_opt solve(prob, LUFactorization()) broken=true
    else
        JET.@test_opt solve(prob, LUFactorization())
    end

    JET.@test_opt solve(prob, GenericLUFactorization())
    JET.@test_opt solve(prob, DiagonalFactorization())
    JET.@test_opt solve(prob, SimpleLUFactorization())
    # JET.@test_opt solve(prob_spd, NormalCholeskyFactorization())
    # JET.@test_opt solve(prob, NormalBunchKaufmanFactorization())
    
    # CholeskyFactorization and SVDFactorization now pass JET tests
    # JET.@test_opt solve(prob_spd, CholeskyFactorization())
    # JET.@test_opt solve(prob, SVDFactorization())
    
    # These tests have runtime dispatch issues on Julia < 1.12
    # Fixed in Julia nightly/pre-release (1.12+)
    if VERSION < v"1.12.0-"
        JET.@test_opt solve(prob, QRFactorization()) broken=true
        JET.@test_opt solve(prob_sym, LDLtFactorization()) broken=true
        JET.@test_opt solve(prob_sym, BunchKaufmanFactorization()) broken=true
    else
        JET.@test_opt solve(prob, QRFactorization())
        JET.@test_opt solve(prob_sym, LDLtFactorization())
        JET.@test_opt solve(prob_sym, BunchKaufmanFactorization())
    end
    JET.@test_opt solve(prob, GenericFactorization()) broken=true
end

@testset "JET Tests for Extension Factorizations" begin
    # RecursiveFactorization.jl extensions
    # JET.@test_opt solve(prob, RFLUFactorization())

    # These tests have runtime dispatch issues on Julia < 1.12
    if VERSION < v"1.12.0-"
        JET.@test_opt solve(prob, FastLUFactorization()) broken=true
        JET.@test_opt solve(prob, FastQRFactorization()) broken=true
    else
        JET.@test_opt solve(prob, FastLUFactorization())
        JET.@test_opt solve(prob, FastQRFactorization())
    end
    
    # Platform-specific factorizations (may not be available on all systems)
    # MKLLUFactorization: Use target_modules to focus JET analysis on LinearSolve code
    # This avoids false positives from Base.show and other stdlib runtime dispatches
    # while still catching real type stability issues in the solver itself
    if @isdefined(MKLLUFactorization)
        JET.@test_opt target_modules=(LinearSolve, SciMLBase) solve(prob, MKLLUFactorization())
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
    # These tests have runtime dispatch issues in SparseArrays stdlib code
    # The dispatches occur in sparse_check_Ti and SparseMatrixCSC constructor
    # These are stdlib issues, not LinearSolve issues
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

    # These tests have Printf runtime dispatch issues in Krylov.jl on Julia < 1.12
    if VERSION < v"1.12.0-"
        JET.@test_opt solve(prob, KrylovJL_GMRES()) broken=true
        JET.@test_opt solve(prob_sym, KrylovJL_MINRES()) broken=true
        JET.@test_opt solve(prob_sym, KrylovJL_MINARES()) broken=true
    else
        JET.@test_opt solve(prob, KrylovJL_GMRES())
        JET.@test_opt solve(prob_sym, KrylovJL_MINRES())
        JET.@test_opt solve(prob_sym, KrylovJL_MINARES())
    end
    
    # Extension Krylov methods (require extensions)
    # KrylovKitJL_CG, KrylovKitJL_GMRES require KrylovKit to be loaded
    # IterativeSolversJL requires IterativeSolvers to be loaded
    # These are tested in their respective extension test suites
end

@testset "JET Tests for Default Solver" begin
    # Test the default solver selection
    # These tests have various runtime dispatch issues in stdlib code:
    # - Dense: Captured variables in appleaccelerate.jl (platform-specific)
    # - Sparse: Runtime dispatch in SparseArrays stdlib, Base.show, etc.
    JET.@test_opt solve(prob) broken=true
    JET.@test_opt solve(prob_sparse) broken=true
end

@testset "JET Tests for creating Dual solutions" begin
    # Make sure there's no runtime dispatch when making solutions of Dual problems
    dual_cache = init(dual_prob, LUFactorization())
    ext = Base.get_extension(LinearSolve, :LinearSolveForwardDiffExt)
    JET.@test_opt ext.linearsolve_dual_solution(
        [1.0, 1.0, 1.0], [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dual_cache)
end

@testset "JET Tests for default algs with DualLinear Problems" begin
    # Test for Default alg choosing for DualLinear Problems
    # These should both produce a LinearCache
    alg = LinearSolve.DefaultLinearSolver(LinearSolve.DefaultAlgorithmChoice.GenericLUFactorization)
    if VERSION < v"1.11"
        JET.@test_opt init(dual_prob, alg) broken=true
        JET.@test_opt init(dual_prob) broken=true
    else
        JET.@test_opt init(dual_prob, alg)
        JET.@test_opt init(dual_prob)
    end
end