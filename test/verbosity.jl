using LinearSolve
using LinearSolve: LinearVerbosity, option_group, group_options, BLISLUFactorization
using SciMLLogging
using Test
@testset "LinearVerbosity Tests" begin
    @testset "Default constructor" begin
        v1 = LinearVerbosity()
        @test v1 isa LinearVerbosity{true}
        @test v1.default_lu_fallback isa SciMLLogging.Warn
        @test v1.KrylovKit_verbosity isa SciMLLogging.Warn
    end

    @testset "Bool constructor" begin
        v2_true = LinearVerbosity(true)
        v2_false = LinearVerbosity(false)
        @test v2_true isa LinearVerbosity{true}
        @test v2_false isa LinearVerbosity{false}
    end

    @testset "VerbosityPreset constructors" begin
        v3_none = LinearVerbosity(SciMLLogging.None())
        v3_all = LinearVerbosity(SciMLLogging.All())
        v3_minimal = LinearVerbosity(SciMLLogging.Minimal())
        v3_standard = LinearVerbosity(SciMLLogging.Standard())
        v3_detailed = LinearVerbosity(SciMLLogging.Detailed())

        @test v3_none isa LinearVerbosity{false}
        @test v3_all isa LinearVerbosity{true}
        @test v3_all.default_lu_fallback isa SciMLLogging.Info
        @test v3_minimal.default_lu_fallback isa SciMLLogging.Error
        @test v3_minimal.KrylovKit_verbosity isa SciMLLogging.Silent
        @test v3_standard isa LinearVerbosity{true}
        @test v3_detailed.KrylovKit_verbosity isa SciMLLogging.Warn
    end

    @testset "Group-level keyword constructors" begin
        v4_error = LinearVerbosity(error_control = SciMLLogging.Error())
        @test v4_error.default_lu_fallback isa SciMLLogging.Error

        v4_numerical = LinearVerbosity(numerical = SciMLLogging.Silent())
        @test v4_numerical.KrylovKit_verbosity isa SciMLLogging.Silent
        @test v4_numerical.using_IterativeSolvers isa SciMLLogging.Silent
        @test v4_numerical.pardiso_verbosity isa SciMLLogging.Silent

        v4_performance = LinearVerbosity(performance = SciMLLogging.Info())
        @test v4_performance.no_right_preconditioning isa SciMLLogging.Info
    end

    @testset "Mixed group and individual settings" begin
        v5_mixed = LinearVerbosity(
            numerical = SciMLLogging.Silent(),
            KrylovKit_verbosity = SciMLLogging.Warn(),
            performance = SciMLLogging.Info()
        )
        # Individual override should take precedence
        @test v5_mixed.KrylovKit_verbosity isa SciMLLogging.Warn
        # Other numerical options should use group setting
        @test v5_mixed.using_IterativeSolvers isa SciMLLogging.Silent
        # Performance group setting should apply
        @test v5_mixed.no_right_preconditioning isa SciMLLogging.Info
    end

    @testset "Individual keyword arguments" begin
        v6_individual = LinearVerbosity(
            default_lu_fallback = SciMLLogging.Error(),
            KrylovKit_verbosity = SciMLLogging.Info(),
            pardiso_verbosity = SciMLLogging.Silent()
        )
        @test v6_individual.default_lu_fallback isa SciMLLogging.Error
        @test v6_individual.KrylovKit_verbosity isa SciMLLogging.Info
        @test v6_individual.pardiso_verbosity isa SciMLLogging.Silent
        # Unspecified options should use defaults
        @test v6_individual.no_right_preconditioning isa SciMLLogging.Warn
    end

    @testset "Group classification functions" begin
        @test option_group(:default_lu_fallback) == :error_control
        @test option_group(:KrylovKit_verbosity) == :numerical
        @test option_group(:no_right_preconditioning) == :performance

        # Test error for unknown option
        @test_throws ErrorException option_group(:unknown_option)
    end

    @testset "Group options function" begin
        v8 = LinearVerbosity(numerical = SciMLLogging.Warn())
        numerical_opts = group_options(v8, :numerical)
        @test numerical_opts isa NamedTuple
        @test :KrylovKit_verbosity in keys(numerical_opts)
        @test :using_IterativeSolvers in keys(numerical_opts)
        @test numerical_opts.KrylovKit_verbosity isa SciMLLogging.Warn

        error_opts = group_options(v8, :error_control)
        @test :default_lu_fallback in keys(error_opts)

        performance_opts = group_options(v8, :performance)
        @test :no_right_preconditioning in keys(performance_opts)

        # Test error for unknown group
        @test_throws ErrorException group_options(v8, :unknown_group)
    end

    @testset "Type parameter consistency" begin
        v_enabled = LinearVerbosity{true}()
        v_disabled = LinearVerbosity{false}()

        @test v_enabled isa LinearVerbosity{true}
        @test v_disabled isa LinearVerbosity{false}

        # Test that the constructors create the right types
        @test LinearVerbosity() isa LinearVerbosity{true}
        @test LinearVerbosity(true) isa LinearVerbosity{true}
        @test LinearVerbosity(false) isa LinearVerbosity{false}
    end

    @testset "Group getproperty access" begin
        v = LinearVerbosity()

        # Test getting groups returns NamedTuples
        error_group = v.error_control
        performance_group = v.performance
        numerical_group = v.numerical

        @test error_group isa NamedTuple
        @test performance_group isa NamedTuple
        @test numerical_group isa NamedTuple

        # Test correct keys are present
        @test :default_lu_fallback in keys(error_group)
        @test :no_right_preconditioning in keys(performance_group)
        @test :KrylovKit_verbosity in keys(numerical_group)
        @test :using_IterativeSolvers in keys(numerical_group)
        @test :pardiso_verbosity in keys(numerical_group)

        # Test values are LogLevel types
        @test error_group.default_lu_fallback isa SciMLLogging.LogLevel
        @test performance_group.no_right_preconditioning isa SciMLLogging.LogLevel
        @test numerical_group.KrylovKit_verbosity isa SciMLLogging.LogLevel

        # Individual field access should still work
        @test v.default_lu_fallback isa SciMLLogging.Warn
        @test v.KrylovKit_verbosity isa SciMLLogging.Warn
    end

    @testset "Group setproperty! setting" begin
        v = LinearVerbosity()

        # Test setting entire error_control group
        v.error_control = SciMLLogging.Error()
        @test v.default_lu_fallback isa SciMLLogging.Error

        # Test setting entire performance group
        v.performance = SciMLLogging.Info()
        @test v.no_right_preconditioning isa SciMLLogging.Info

        # Test setting entire numerical group
        v.numerical = SciMLLogging.Silent()
        @test v.KrylovKit_verbosity isa SciMLLogging.Silent
        @test v.using_IterativeSolvers isa SciMLLogging.Silent
        @test v.pardiso_verbosity isa SciMLLogging.Silent
        @test v.HYPRE_verbosity isa SciMLLogging.Silent

        # Test that other groups aren't affected
        @test v.default_lu_fallback isa SciMLLogging.Error  # error_control unchanged
        @test v.no_right_preconditioning isa SciMLLogging.Info  # performance unchanged

        # Test individual setting still works after group setting
        v.KrylovKit_verbosity = SciMLLogging.Warn()
        @test v.KrylovKit_verbosity isa SciMLLogging.Warn
        # Other numerical options should still be Silent
        @test v.using_IterativeSolvers isa SciMLLogging.Silent
    end

    @testset "Group setproperty! error handling" begin
        v = LinearVerbosity()

        # Test error for invalid group value type
        @test_throws ErrorException v.error_control = "invalid"
        @test_throws ErrorException v.performance = 123
        @test_throws ErrorException v.numerical = :invalid

        # Test error for invalid individual option type
        @test_throws ErrorException v.KrylovKit_verbosity = "invalid"
        @test_throws ErrorException v.default_lu_fallback = 123
    end

    @testset "getproperty and setproperty! consistency" begin
        v = LinearVerbosity()

        # Set a group and verify getproperty reflects the change
        v.numerical = SciMLLogging.Error()
        numerical_group = v.numerical

        @test all(x -> x isa SciMLLogging.Error, values(numerical_group))

        # Set individual option and verify both individual and group access work
        v.KrylovKit_verbosity = SciMLLogging.Info()
        @test v.KrylovKit_verbosity isa SciMLLogging.Info

        updated_numerical = v.numerical
        @test updated_numerical.KrylovKit_verbosity isa SciMLLogging.Info
        # Other numerical options should still be Error
        @test updated_numerical.using_IterativeSolvers isa SciMLLogging.Error
    end
end


A = [1.0 0 0 0
     0 1 0 0
     0 0 1 0
     0 0 0 0]
b = rand(4)
prob = LinearProblem(A, b)

@test_logs (:warn,
    "LU factorization failed, falling back to QR factorization. `A` is potentially rank-deficient.") solve(
    prob,
    verbose = LinearVerbosity(default_lu_fallback = SciMLLogging.Warn()))

@test_logs (:warn,
    "LU factorization failed, falling back to QR factorization. `A` is potentially rank-deficient.") solve(
    prob, verbose = true)

@test_logs min_level=SciMLLogging.Logging.Warn solve(prob, verbose = false)

@test_logs (:info,
    "LU factorization failed, falling back to QR factorization. `A` is potentially rank-deficient.") solve(
    prob,
    verbose = LinearVerbosity(default_lu_fallback = SciMLLogging.Info()))

verb = LinearVerbosity(default_lu_fallback = SciMLLogging.Warn())

@test_logs (:warn,
    "LU factorization failed, falling back to QR factorization. `A` is potentially rank-deficient.") solve(
    prob,
    verbose = verb)

verb.default_lu_fallback = SciMLLogging.Info()

@test_logs (:info,
    "LU factorization failed, falling back to QR factorization. `A` is potentially rank-deficient.") solve(
    prob,
    verbose = verb)

@testset "BLAS Return Code Interpretation" begin
    # Test interpretation of various BLAS return codes
    @testset "Return Code Interpretation" begin
        # Test successful operation
        category, message, details = LinearSolve.interpret_blas_code(:dgetrf, 0)
        @test category == :success
        @test message == "Operation completed successfully"

        # Test invalid argument
        category, message, details = LinearSolve.interpret_blas_code(:dgetrf, -3)
        @test category == :invalid_argument
        @test occursin("Argument 3", details)

        # Test singular matrix in LU
        category, message, details = LinearSolve.interpret_blas_code(:dgetrf, 2)
        @test category == :singular_matrix
        @test occursin("U(2,2)", details)

        # Test not positive definite in Cholesky
        category, message, details = LinearSolve.interpret_blas_code(:dpotrf, 3)
        @test category == :not_positive_definite
        @test occursin("minor of order 3", details)

        # Test SVD convergence failure
        category, message, details = LinearSolve.interpret_blas_code(:dgesvd, 5)
        @test category == :convergence_failure
        @test occursin("5 off-diagonal", details)
    end

    @testset "BLAS Operation Info" begin
        # Test getting operation info without condition number
        A = rand(10, 10)
        b = rand(10)

        # Test with condition_number disabled (default)
        info = LinearSolve.get_blas_operation_info(:dgetrf, A, b)

        @test info[:matrix_size] == (10, 10)
        @test info[:element_type] == Float64
        @test !haskey(info, :condition_number)  # Should not compute by default
        @test info[:memory_usage_MB] >= 0  # Memory can be 0 for very small matrices

        # Test with condition number computation enabled via verbosity
        verbose_with_cond = LinearVerbosity(condition_number = SciMLLogging.Info())
        info_with_cond = LinearSolve.get_blas_operation_info(
            :dgetrf, A, b, condition = !isa(verbose_with_cond.condition_number, SciMLLogging.Silent))
        @test haskey(info_with_cond, :condition_number)
    end

    @testset "Error Categories" begin
        # Test different error categories are properly identified
        test_cases = [
            (:dgetrf, 1, :singular_matrix),
            (:dpotrf, 2, :not_positive_definite),
            (:dgeqrf, 3, :numerical_issue),
            (:dgesdd, 4, :convergence_failure),
            (:dsyev, 5, :convergence_failure),
            (:dsytrf, 6, :singular_matrix),
            (:dgetrs, 1, :unexpected_error),
            (:unknown_func, 1, :unknown_error)
        ]

        for (func, code, expected_category) in test_cases
            category, _, _ = LinearSolve.interpret_blas_code(func, code)
            @test category == expected_category
        end
    end
end

# Try to load BLIS extension
try
    using blis_jll, LAPACK_jll
catch LoadError
    # BLIS dependencies not available, tests will be skipped
end

@testset "BLIS Verbosity Integration Tests" begin
    @testset "BLIS solver with verbosity logging" begin
        # Test basic BLIS solver functionality with verbosity
        if Base.get_extension(LinearSolve, :LinearSolveBLISExt) == nothing
            # Only test if BLIS is available
            @info "Skipping BLIS tests - BLIS not available"
        else
            # Test successful solve with success logging enabled
            A_good = [2.0 1.0; 1.0 2.0]
            b_good = [3.0, 4.0]
            prob_good = LinearProblem(A_good, b_good)

            verbose_success = LinearVerbosity(
                blas_success = SciMLLogging.Info(),
                blas_errors = SciMLLogging.Silent(),
                blas_info = SciMLLogging.Silent()
            )

            @test_logs (:info, r"BLAS LU factorization.*completed successfully") solve(
                prob_good, BLISLUFactorization(); verbose = verbose_success)

            # Test singular matrix with error logging
            A_singular = [1.0 2.0; 2.0 4.0]
            b_singular = [1.0, 2.0]
            prob_singular = LinearProblem(A_singular, b_singular)

            verbose_errors = LinearVerbosity(
                blas_errors = SciMLLogging.Warn(),
                blas_success = SciMLLogging.Silent(),
                blas_info = SciMLLogging.Silent()
            )

            @test_logs (:warn, r"BLAS/LAPACK.*Matrix is singular") solve(
                prob_singular, BLISLUFactorization(); verbose = verbose_errors)

            # Test with info logging enabled
            verbose_info = LinearVerbosity(
                blas_info = SciMLLogging.Info(),
                blas_errors = SciMLLogging.Info(),
                blas_success = SciMLLogging.Silent()
            )

            @test_logs (:info, r"BLAS/LAPACK.*Matrix is singular") solve(
                prob_singular, BLISLUFactorization(); verbose = verbose_info)

            # Test with all BLAS logging disabled - should produce no logs
            verbose_silent = LinearVerbosity(
                blas_errors = SciMLLogging.Silent(),
                blas_invalid_args = SciMLLogging.Silent(),
                blas_info = SciMLLogging.Silent(),
                blas_success = SciMLLogging.Silent()
            )

            @test_logs min_level=SciMLLogging.Logging.Warn solve(
                prob_singular, BLISLUFactorization(); verbose = verbose_silent)

            # Test condition number logging if enabled
            verbose_with_cond = LinearVerbosity(
                condition_number = SciMLLogging.Info(),
                blas_success = SciMLLogging.Info(),
                blas_errors = SciMLLogging.Silent()
            )

            @test_logs (:info, r"Matrix condition number:.*for.*matrix") match_mode=:any solve(
                prob_good, BLISLUFactorization(); verbose = verbose_with_cond)
        end
    end
end