using LinearSolve, ForwardDiff, ForwardDiff, RecursiveFactorization, LinearAlgebra, SparseArrays, Test
using JET

# Loaded for the extension module, matching test/qa/allocations.jl: without both
# JLLs the BLIS extension never loads and BLISLUFactorization() throws.
if Sys.islinux()
    import LAPACK_jll, blis_jll
end

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
    return (
        A = [
            p[1] p[2] + 1 p[2]^3;
            3 * p[1] p[1] + 5 p[2] * p[1] - 4;
            p[2]^2 9 * p[1] p[2]
        ],
        b = [p[1] + 1, p[2] * 2, p[1]^2],
    )
end

A, b = h([ForwardDiff.Dual(5.0, 1.0, 0.0), ForwardDiff.Dual(5.0, 0.0, 1.0)])

dual_prob = LinearProblem(A, b)

# Dual problem set up
function h(p)
    return (
        A = [
            p[1] p[2] + 1 p[2]^3;
            3 * p[1] p[1] + 5 p[2] * p[1] - 4;
            p[2]^2 9 * p[1] p[2]
        ],
        b = [p[1] + 1, p[2] * 2, p[1]^2],
    )
end

A, b = h([ForwardDiff.Dual(5.0, 1.0, 0.0), ForwardDiff.Dual(5.0, 0.0, 1.0)])

dual_prob = LinearProblem(A, b)

@testset "JET Tests for Dense Factorizations" begin
    # Working tests - these pass JET optimization checks
    JET.@test_opt init(prob, nothing)

    # LUFactorization and GenericLUFactorization have runtime dispatch in
    # LinearAlgebra.mul! (used by residualsafety check) on Julia < 1.11
    if VERSION < v"1.11"
        JET.@test_opt solve(prob, LUFactorization()) broken = true
        JET.@test_opt solve(prob, GenericLUFactorization()) broken = true
    else
        JET.@test_opt solve(prob, LUFactorization())
        JET.@test_opt solve(prob, GenericLUFactorization())
    end
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
        JET.@test_opt solve(prob, QRFactorization()) broken = true
        JET.@test_opt solve(prob_sym, LDLtFactorization()) broken = true
        JET.@test_opt solve(prob_sym, BunchKaufmanFactorization()) broken = true
    else
        JET.@test_opt solve(prob, QRFactorization())
        JET.@test_opt solve(prob_sym, LDLtFactorization())
        JET.@test_opt solve(prob_sym, BunchKaufmanFactorization())
    end
    JET.@test_opt solve(prob, GenericFactorization()) broken = true
end

@testset "JET Tests for Extension Factorizations" begin
    # RecursiveFactorization.jl extensions
    # JET.@test_opt solve(prob, RFLUFactorization())

    # These tests have runtime dispatch issues on Julia < 1.12
    if VERSION < v"1.12.0-"
        JET.@test_opt solve(prob, FastLUFactorization()) broken = true
        JET.@test_opt solve(prob, FastQRFactorization()) broken = true
    else
        JET.@test_opt solve(prob, FastLUFactorization())
        JET.@test_opt solve(prob, FastQRFactorization())
    end

    # Platform-specific factorizations (may not be available on all systems)
    # MKLLUFactorization: Use target_modules to focus JET analysis on LinearSolve code
    # This avoids false positives from Base.show and other stdlib runtime dispatches
    # while still catching real type stability issues in the solver itself
    if @isdefined(MKLLUFactorization)
        JET.@test_opt target_modules = (LinearSolve, SciMLBase) solve(prob, MKLLUFactorization())
    end

    if Sys.isapple() && @isdefined(AppleAccelerateLUFactorization)
        JET.@test_opt solve(prob, AppleAccelerateLUFactorization()) broken = true
    end

    # CUDA/Metal factorizations (only test if CUDA/Metal are loaded)
    # CudaOffloadFactorization requires CUDA to be loaded, skip if not available
    # Metal is only available on Apple platforms
    if Sys.isapple() && @isdefined(MetalLUFactorization)
        JET.@test_opt solve(prob, MetalLUFactorization()) broken = true
    end
    # BLISLUFactorization is exported unconditionally, so @isdefined does not
    # tell us whether it can be constructed; the extension has to be loaded.
    if Base.get_extension(LinearSolve, :LinearSolveBLISExt) !== nothing
        JET.@test_opt solve(prob, BLISLUFactorization()) broken = true
    end
end

@testset "JET Tests for Sparse Factorizations" begin
    # These tests have runtime dispatch issues in SparseArrays stdlib code
    # The dispatches occur in sparse_check_Ti and SparseMatrixCSC constructor
    # These are stdlib issues, not LinearSolve issues
    JET.@test_opt solve(prob_sparse, UMFPACKFactorization()) broken = true
    # Passes since the 5.0 lightweight solution: with the cache no longer
    # captured in the returned LinearSolution, the KLU solve is dispatch-clean.
    #
    # Except on the LTS. `@SciMLMessage` in the failure branches expands to
    # `Logging.@logmsg`, and on 1.10 the `Base.CoreLogging` path it enters
    # reaches `Base.typejoin`, which is itself a runtime dispatch there. That
    # makes *any* solve able to emit a log fail `@test_opt` on 1.10, regardless
    # of anything LinearSolve does; 1.11 carries the Base fix. Measured on this
    # assertion: 2 reports on 1.10.11, 0 on 1.12.6, and the 1.10 reports name
    # only `typejoin`/`CoreLogging`/`_emit_log`, none of the KLU internals that
    # #1148 and #1163 dealt with. See SciML/LinearSolve.jl#1190.
    JET.@test_opt solve(prob_sparse, KLUFactorization()) broken = VERSION < v"1.11"
    JET.@test_opt solve(prob_sparse_spd, CHOLMODFactorization()) broken = true

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
        JET.@test_opt solve(prob, KrylovJL_GMRES()) broken = true
        JET.@test_opt solve(prob_sym, KrylovJL_MINRES()) broken = true
        JET.@test_opt solve(prob_sym, KrylovJL_MINARES()) broken = true
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
    # Julia 1.10 reports runtime dispatch through stdlib and Krylov fallback paths.
    #
    # `target_modules` for the same reason as the MKLLUFactorization test above:
    # JET analyzes every branch of the generated DefaultLinearSolver `solve!`
    # regardless of which one would run, and the MKL branch's failure-logging
    # path (`get_blas_operation_info`'s `string(::Type)`) dispatches inside
    # Base's `show` machinery. Those frames are stdlib-internal; real dispatch
    # in LinearSolve/SciMLBase code is still reported. This was masked until
    # the KLU getproperty fix (#1148) let the suite get past the sparse testset.
    JET.@test_opt target_modules = (LinearSolve, SciMLBase) solve(prob) broken =
        VERSION < v"1.12.0-"
    # Sparse has runtime dispatch in SparseArrays stdlib, Base.show, etc.
    JET.@test_opt solve(prob_sparse) broken = true
end

@testset "JET Tests for creating Dual solutions" begin
    # Make sure there's no runtime dispatch when making solutions of Dual problems
    dual_cache = init(dual_prob, LUFactorization())
    ext = Base.get_extension(LinearSolve, :LinearSolveForwardDiffExt)
    JET.@test_opt ext.linearsolve_dual_solution(
        [1.0, 1.0, 1.0], [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dual_cache
    )
end

@testset "JET Tests for default algs with DualLinear Problems" begin
    # Test for Default alg choosing for DualLinear Problems
    # These should both produce a LinearCache
    alg = LinearSolve.DefaultLinearSolver(LinearSolve.DefaultAlgorithmChoice.GenericLUFactorization)
    if VERSION < v"1.11"
        JET.@test_opt init(dual_prob, alg) broken = true
        JET.@test_opt init(dual_prob) broken = true
    else
        JET.@test_opt init(dual_prob, alg)
        JET.@test_opt init(dual_prob)
    end
end

# Concrete-return-type QA for `solve!(cache)`. Guards against the regression
# where `solve!(cache)` through `DefaultLinearSolver` returned
# `LinearSolution{_A, _B, _C, _D, DefaultLinearSolver, _E, _F} where {...}`
# (a UnionAll over 6 free type parameters) instead of a concrete LinearSolution.
_solve_alg(A, b, alg) = solve!(init(LinearProblem(A, b), alg))
_solve_default(A, b) = solve!(init(LinearProblem(A, b)))

@testset "solve!(cache) returns concrete LinearSolution — default solver" begin
    # Headline case: `solve!(cache)` after `init(LinearProblem(A, b))` must not
    # return a UnionAll-typed LinearSolution. Was broken by the
    # `_default_lu_solve_with_fallback`/`_do_qr_fallback` helpers reading
    # `sol.u`/`sol.resid`/`sol.cache`/`sol.stats` from an inner `sol` whose
    # rettype got capped to `Any` during precompile.
    rt = Core.Compiler.return_type(
        _solve_default, Tuple{Matrix{Float64}, Vector{Float64}}
    )
    @test isconcretetype(rt)
    @test rt <: LinearSolve.SciMLBase.LinearSolution{Float64, 1, Vector{Float64}}
end

@testset "solve!(cache) is concrete for each algorithm" begin
    algs_concrete = (
        LUFactorization(),
        GenericLUFactorization(),
        QRFactorization(LinearAlgebra.ColumnNorm()),
        QRFactorization(LinearAlgebra.NoPivot()),
        DiagonalFactorization(),
        SVDFactorization(),
        CholeskyFactorization(),
        NormalCholeskyFactorization(),
    )
    for alg in algs_concrete
        @testset "$(nameof(typeof(alg)))" begin
            rt = Core.Compiler.return_type(
                _solve_alg,
                Tuple{Matrix{Float64}, Vector{Float64}, typeof(alg)}
            )
            @test isconcretetype(rt)
        end
    end

    # Known unrelated inference issues — tracked separately, not what this
    # group is guarding against.
    algs_broken = (
        BunchKaufmanFactorization(),
        LDLtFactorization(),
    )
    for alg in algs_broken
        @testset "$(nameof(typeof(alg))) (broken)" begin
            rt = Core.Compiler.return_type(
                _solve_alg,
                Tuple{Matrix{Float64}, Vector{Float64}, typeof(alg)}
            )
            @test_broken isconcretetype(rt)
        end
    end
end

@testset "JET Tests for SupernodalLU" begin
    # The dense block caches are concretely typed, so the repeated-use paths -
    # solve and numeric refactorization, i.e. the ODE/Newton workload - must be
    # free of runtime dispatch.  Analysis is scoped to the solver module, as
    # elsewhere in this file, so Base error-path noise (show/AssertionError
    # reachable from sparse indexing) does not mask our own code.
    SNLU_MOD = LinearSolve.SupernodalLU
    n = 60
    A_snlu = SparseArrays.spdiagm(
        0 => fill(4.0, n), 1 => fill(-1.0, n - 1), -1 => fill(-1.0, n - 1)
    )
    b_snlu = ones(n)
    F_snlu = SNLU_MOD.snlu(A_snlu)
    x_snlu = similar(b_snlu)

    # The solve path never touches the block caches, so it must be free of
    # runtime dispatch.
    JET.@test_opt target_modules = (SNLU_MOD,) SNLU_MOD.solve!(x_snlu, F_snlu, b_snlu)
    JET.@test_opt target_modules = (SNLU_MOD,) SNLU_MOD._solve_panels!(x_snlu, F_snlu)

    # Factorization carries exactly one dispatch per *cached* supernode: the
    # `_cache_lu!` barrier, whose cache argument is deliberately untyped (see
    # the `bcaches` field docs - putting the cache type in the factorization
    # type breaks `@inferred init(prob, nothing)` for sparse matrices, because
    # the sparse default holds a SupernodalLUFactor in its cacheval).  Tens of
    # calls against hundreds of ms of GEMM; the alternative costs far more.
    JET.@test_opt target_modules = (SNLU_MOD,) SNLU_MOD.snlu!(F_snlu, A_snlu) broken = true
    JET.@test_opt target_modules = (SNLU_MOD,) SNLU_MOD.snlu(A_snlu) broken = true

    # The factorization object itself is concrete, and its block-cache vector
    # is a two-member union rather than Any.
    @test isconcretetype(typeof(F_snlu))
    @test typeof(F_snlu) === SNLU_MOD.SupernodalLUFactor{Float64, Int}

    # The factorization type itself is fully concrete and inferable from its
    # inputs, which is what the sparse default solver depends on.
    @test isconcretetype(
        Core.Compiler.return_type(SNLU_MOD.snlu, Tuple{typeof(A_snlu)})
    )
end
