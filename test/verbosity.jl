using LinearSolve
using LinearSolve: LinearVerbosity, option_group, group_options, BLISLUFactorization,
    __appleaccelerate_isavailable, __mkl_isavailable, __openblas_isavailable
using SciMLLogging
using Test

@testset "LinearVerbosity Tests" begin
    @testset "Default constructor" begin
        v1 = LinearVerbosity()
        @test v1 isa LinearVerbosity
        @test v1.default_lu_fallback isa SciMLLogging.Silent
        @test v1.KrylovKit_verbosity == SciMLLogging.CustomLevel(1)
    end
    @testset "LinearVerbosity constructors" begin
        v3_none = LinearVerbosity(SciMLLogging.None())
        v3_all = LinearVerbosity(SciMLLogging.All())
        v3_minimal = LinearVerbosity(SciMLLogging.Minimal())
        v3_standard = LinearVerbosity(SciMLLogging.Standard())
        v3_detailed = LinearVerbosity(SciMLLogging.Detailed())

        @test v3_all.default_lu_fallback isa SciMLLogging.WarnLevel
        @test v3_minimal.default_lu_fallback isa SciMLLogging.Silent
        @test v3_minimal.KrylovKit_verbosity isa SciMLLogging.Silent
        @test v3_detailed.KrylovKit_verbosity == SciMLLogging.CustomLevel(2)
    end

    @testset "Group-level keyword constructors" begin
        v4_error = LinearVerbosity(error_control = ErrorLevel())
        @test v4_error.default_lu_fallback isa SciMLLogging.ErrorLevel

        v4_numerical = LinearVerbosity(numerical = Silent())
        @test v4_numerical.KrylovKit_verbosity isa SciMLLogging.Silent
        @test v4_numerical.using_IterativeSolvers isa SciMLLogging.Silent
        @test v4_numerical.pardiso_verbosity isa SciMLLogging.Silent

        v4_performance = LinearVerbosity(performance = InfoLevel())
        @test v4_performance.no_right_preconditioning isa SciMLLogging.InfoLevel
    end

    @testset "Mixed group and individual settings" begin
        v5_mixed = LinearVerbosity(
            numerical = Silent(),
            KrylovKit_verbosity = WarnLevel(),
            performance = InfoLevel()
        )
        # Individual override should take precedence
        @test v5_mixed.KrylovKit_verbosity isa SciMLLogging.WarnLevel
        # Other numerical options should use group setting
        @test v5_mixed.using_IterativeSolvers isa SciMLLogging.Silent
        # Performance group setting should apply
        @test v5_mixed.no_right_preconditioning isa SciMLLogging.InfoLevel
    end

    @testset "Individual keyword arguments" begin
        v6_individual = LinearVerbosity(
            default_lu_fallback = ErrorLevel(),
            KrylovKit_verbosity = InfoLevel(),
            pardiso_verbosity = Silent()
        )
        @test v6_individual.default_lu_fallback isa SciMLLogging.ErrorLevel
        @test v6_individual.KrylovKit_verbosity isa SciMLLogging.InfoLevel
        @test v6_individual.pardiso_verbosity isa SciMLLogging.Silent
        # Unspecified options should use defaults
        @test v6_individual.no_right_preconditioning isa SciMLLogging.Silent
    end

    @testset "Group classification functions" begin
        @test option_group(:default_lu_fallback) == :error_control
        @test option_group(:KrylovKit_verbosity) == :numerical
        @test option_group(:no_right_preconditioning) == :performance

        # Test error for unknown option
        @test_throws ErrorException option_group(:unknown_option)
    end

    @testset "Group options function" begin
        v8 = LinearVerbosity(numerical = WarnLevel())
        numerical_opts = group_options(v8, :numerical)
        @test numerical_opts isa NamedTuple
        @test :KrylovKit_verbosity in keys(numerical_opts)
        @test :using_IterativeSolvers in keys(numerical_opts)
        @test numerical_opts.KrylovKit_verbosity isa SciMLLogging.WarnLevel

        error_opts = group_options(v8, :error_control)
        @test :default_lu_fallback in keys(error_opts)

        performance_opts = group_options(v8, :performance)
        @test :no_right_preconditioning in keys(performance_opts)

        # Test error for unknown group
        @test_throws ErrorException group_options(v8, :unknown_group)
    end
end


@testset "LinearVerbosity Logs Tests" begin
    A = [1.0 0 0 0
         0 1 0 0
         0 0 1 0
         0 0 0 0]
    b = rand(4)
    prob = LinearProblem(A, b)

    @test_logs (:warn,
        "LU factorization failed, falling back to QR factorization. `A` is potentially rank-deficient.") solve(
        prob,
        verbose = LinearVerbosity(default_lu_fallback = WarnLevel()))

    @test_logs (:warn, r"Using `true` or `false` for `verbose` is being deprecated") match_mode=:any min_level=SciMLLogging.Logging.Warn solve(prob, verbose = false)

    @test_logs (:info,
        "LU factorization failed, falling back to QR factorization. `A` is potentially rank-deficient.") solve(
        prob,
        verbose = LinearVerbosity(default_lu_fallback = InfoLevel()))

    verb = LinearVerbosity(default_lu_fallback = WarnLevel())

    @test_logs (:warn,
        "LU factorization failed, falling back to QR factorization. `A` is potentially rank-deficient.") solve(
        prob,
        verbose = verb)

end

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
        verbose_with_cond = LinearVerbosity(condition_number = InfoLevel())
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
        if Base.get_extension(LinearSolve, :LinearSolveBLISExt) === nothing
            # Only test if BLIS is available
            @info "Skipping BLIS tests - BLIS not available"
        else
            # Test successful solve with success logging enabled
            A_good = [2.0 1.0; 1.0 2.0]
            b_good = [3.0, 4.0]
            prob_good = LinearProblem(A_good, b_good)

            verbose_success = LinearVerbosity(
                blas_success = InfoLevel(),
                blas_errors = Silent(),
                blas_info = Silent()
            )

            @test_logs (:info, r"BLAS LU factorization.*completed successfully") solve(
                prob_good, BLISLUFactorization(); verbose = verbose_success)

            # Test singular matrix with error logging
            A_singular = [1.0 2.0; 2.0 4.0]
            b_singular = [1.0, 2.0]
            prob_singular = LinearProblem(A_singular, b_singular)

            verbose_errors = LinearVerbosity(
                blas_errors = WarnLevel(),
                blas_success = Silent(),
                blas_info = Silent()
            )

            @test_logs (:warn, r"BLAS/LAPACK.*Matrix is singular") solve(
                prob_singular, BLISLUFactorization(); verbose = verbose_errors)

            # Test with info logging enabled
            verbose_info = LinearVerbosity(
                blas_info = InfoLevel(),
                blas_errors = InfoLevel(),
                blas_success = Silent()
            )

            @test_logs (:info, r"BLAS/LAPACK.*Matrix is singular") solve(
                prob_singular, BLISLUFactorization(); verbose = verbose_info)

            # Test with all BLAS logging disabled - should produce no logs
            verbose_silent = LinearVerbosity(
                blas_errors = Silent(),
                blas_invalid_args = Silent(),
                blas_info = Silent(),
                blas_success = Silent(),
                solver_failure = Silent()
            )

            @test_logs min_level=SciMLLogging.Logging.Warn solve(
                prob_singular, BLISLUFactorization(); verbose = verbose_silent)

            # Test condition number logging if enabled
            verbose_with_cond = LinearVerbosity(
                condition_number = InfoLevel(),
                blas_success = InfoLevel(),
                blas_errors = Silent()
            )

            @test_logs (:info, r"Matrix condition number:.*for.*matrix") match_mode=:any solve(
                prob_good, BLISLUFactorization(); verbose = verbose_with_cond)
        end
    end
end

@testset "OpenBLAS Verbosity Integration Tests" begin
    @testset "OpenBLAS solver with verbosity logging" begin
        # Test basic OpenBLAS solver functionality with verbosity
        if __openblas_isavailable()
            # Test successful solve with success logging enabled
            A_good = [2.0 1.0; 1.0 2.0]
            b_good = [3.0, 4.0]
            prob_good = LinearProblem(A_good, b_good)

            verbose_success = LinearVerbosity(
                blas_success = InfoLevel(),
                blas_errors = Silent(),
                blas_info = Silent()
            )

            @test_logs (:info, r"BLAS LU factorization.*completed successfully") solve(
                prob_good, OpenBLASLUFactorization(); verbose = verbose_success)

            # Test singular matrix with error logging
            A_singular = [1.0 2.0; 2.0 4.0]
            b_singular = [1.0, 2.0]
            prob_singular = LinearProblem(A_singular, b_singular)

            verbose_errors = LinearVerbosity(
                blas_errors = WarnLevel(),
                blas_success = Silent(),
                blas_info = Silent()
            )

            @test_logs (:warn, r"BLAS/LAPACK.*Matrix is singular") match_mode=:any solve(
                prob_singular, OpenBLASLUFactorization(); verbose = verbose_errors)

            # Test with info logging enabled
            verbose_info = LinearVerbosity(
                blas_info = InfoLevel(),
                blas_errors = InfoLevel(),
                blas_success = Silent()
            )

            @test_logs (:info, r"BLAS/LAPACK.*Matrix is singular") match_mode=:any solve(
                prob_singular, OpenBLASLUFactorization(); verbose = verbose_info)

            # Test with all BLAS logging disabled - should produce no logs
            verbose_silent = LinearVerbosity(
                blas_errors = Silent(),
                blas_invalid_args = Silent(),
                blas_info = Silent(),
                blas_success = Silent(),
                solver_failure = Silent()
            )

            @test_logs min_level=SciMLLogging.Logging.Warn solve(
                prob_singular, OpenBLASLUFactorization(); verbose = verbose_silent)

            # Test condition number logging if enabled
            verbose_with_cond = LinearVerbosity(
                condition_number = InfoLevel(),
                blas_success = InfoLevel(),
                blas_errors = Silent()
            )

            @test_logs (:info, r"Matrix condition number:.*for.*matrix") match_mode=:any solve(
                prob_good, OpenBLASLUFactorization(); verbose = verbose_with_cond)
        else
            @info "Skipping OpenBLAS tests - OpenBLAS not available"
        end
    end
end

@testset "AppleAccelerate Verbosity Integration Tests" begin
    @testset "AppleAccelerate solver with verbosity logging" begin
        # Test basic AppleAccelerate solver functionality with verbosity
        if __appleaccelerate_isavailable()
            # Test successful solve with success logging enabled
            A_good = [2.0 1.0; 1.0 2.0]
            b_good = [3.0, 4.0]
            prob_good = LinearProblem(A_good, b_good)

            verbose_success = LinearVerbosity(
                blas_success = InfoLevel(),
                blas_errors = Silent(),
                blas_info = Silent()
            )

            @test_logs (:info, r"BLAS LU factorization.*completed successfully") solve(
                prob_good, AppleAccelerateLUFactorization(); verbose = verbose_success)

            # Test singular matrix with error logging
            A_singular = [1.0 2.0; 2.0 4.0]
            b_singular = [1.0, 2.0]
            prob_singular = LinearProblem(A_singular, b_singular)

            verbose_errors = LinearVerbosity(
                blas_errors = WarnLevel(),
                blas_success = Silent(),
                blas_info = Silent()
            )

            @test_logs (:warn, r"BLAS/LAPACK.*Matrix is singular") solve(
                prob_singular, AppleAccelerateLUFactorization(); verbose = verbose_errors)

            # Test with info logging enabled
            verbose_info = LinearVerbosity(
                blas_info = InfoLevel(),
                blas_errors = InfoLevel(),
                blas_success = Silent()
            )

            @test_logs (:info, r"BLAS/LAPACK.*Matrix is singular") solve(
                prob_singular, AppleAccelerateLUFactorization(); verbose = verbose_info)

            # Test with all BLAS logging disabled - should produce no logs
            verbose_silent = LinearVerbosity(
                blas_errors = Silent(),
                blas_invalid_args = Silent(),
                blas_info = Silent(),
                blas_success = Silent(),
                solver_failure = Silent()
            )

            @test_logs min_level=SciMLLogging.Logging.Warn solve(
                prob_singular, AppleAccelerateLUFactorization(); verbose = verbose_silent)

            # Test condition number logging if enabled
            verbose_with_cond = LinearVerbosity(
                condition_number = InfoLevel(),
                blas_success = InfoLevel(),
                blas_errors = Silent()
            )

            @test_logs (:info, r"Matrix condition number:.*for.*matrix") match_mode=:any solve(
                prob_good, AppleAccelerateLUFactorization(); verbose = verbose_with_cond)
        else
            @info "Skipping AppleAccelerate tests - AppleAccelerate not available"
        end
    end
end

@testset "MKL Verbosity Integration Tests" begin
    @testset "MKL solver with verbosity logging" begin
        # Test basic MKL solver functionality with verbosity
        if __mkl_isavailable()
            # Test successful solve with success logging enabled
            A_good = [2.0 1.0; 1.0 2.0]
            b_good = [3.0, 4.0]
            prob_good = LinearProblem(A_good, b_good)

            verbose_success = LinearVerbosity(
                blas_success = InfoLevel(),
                blas_errors = Silent(),
                blas_info = Silent()
            )

            @test_logs (:info, r"BLAS LU factorization.*completed successfully") match_mode=:any solve(
                prob_good, MKLLUFactorization(); verbose = verbose_success)

            # Test singular matrix with error logging
            A_singular = [1.0 2.0; 2.0 4.0]
            b_singular = [1.0, 2.0]
            prob_singular = LinearProblem(A_singular, b_singular)

            verbose_errors = LinearVerbosity(
                blas_errors = WarnLevel(),
                blas_success = Silent(),
                blas_info = Silent()
            )

            @test_logs (:warn, r"BLAS/LAPACK.*Matrix is singular") match_mode=:any solve(
                prob_singular, MKLLUFactorization(); verbose = verbose_errors)

            # Test with info logging enabled
            verbose_info = LinearVerbosity(
                blas_info = InfoLevel(),
                blas_errors = InfoLevel(),
                blas_success = Silent()
            )

            @test_logs (:info, r"BLAS/LAPACK.*Matrix is singular") match_mode=:any solve(
                prob_singular, MKLLUFactorization(); verbose = verbose_info)

            # Test with all BLAS logging disabled - should produce no logs
            verbose_silent = LinearVerbosity(
                blas_errors = Silent(),
                blas_invalid_args = Silent(),
                blas_info = Silent(),
                blas_success = Silent(),
                solver_failure = Silent()
            )

            @test_logs min_level=SciMLLogging.Logging.Warn match_mode=:any solve(
                prob_singular, MKLLUFactorization(); verbose = verbose_silent)

            # Test condition number logging if enabled
            verbose_with_cond = LinearVerbosity(
                condition_number = InfoLevel(),
                blas_success = InfoLevel(),
                blas_errors = Silent()
            )

            @test_logs (:info, r"Matrix condition number:.*for.*matrix") match_mode=:any solve(
                prob_good, MKLLUFactorization(); verbose = verbose_with_cond)
        else
            @info "Skipping MKL tests - MKL not available"
        end
    end
end