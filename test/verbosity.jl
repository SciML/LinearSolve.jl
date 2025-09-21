using LinearSolve
using LinearSolve: LinearVerbosity, option_group, group_options
using SciMLLogging: SciMLLogging, Verbosity
using Test
@testset "LinearVerbosity Tests" begin
    @testset "Default constructor" begin
        v1 = LinearVerbosity()
        @test v1 isa LinearVerbosity{true}
        @test v1.default_lu_fallback isa SciMLLogging.Verbosity.Warn
        @test v1.KrylovKit_verbosity isa SciMLLogging.Verbosity.Warn
    end

    @testset "Bool constructor" begin
        v2_true = LinearVerbosity(true)
        v2_false = LinearVerbosity(false)
        @test v2_true isa LinearVerbosity{true}
        @test v2_false isa LinearVerbosity{false}
    end

    @testset "VerbosityPreset constructors" begin
        v3_none = LinearVerbosity(SciMLLogging.Verbosity.None())
        v3_all = LinearVerbosity(SciMLLogging.Verbosity.All())
        v3_minimal = LinearVerbosity(SciMLLogging.Verbosity.Minimal())
        v3_standard = LinearVerbosity(SciMLLogging.Verbosity.Standard())
        v3_detailed = LinearVerbosity(SciMLLogging.Verbosity.Detailed())

        @test v3_none isa LinearVerbosity{false}
        @test v3_all isa LinearVerbosity{true}
        @test v3_all.default_lu_fallback isa SciMLLogging.Verbosity.Info
        @test v3_minimal.default_lu_fallback isa SciMLLogging.Verbosity.Error
        @test v3_minimal.KrylovKit_verbosity isa SciMLLogging.Verbosity.Silent
        @test v3_standard isa LinearVerbosity{true}
        @test v3_detailed.KrylovKit_verbosity isa SciMLLogging.Verbosity.Warn
    end

    @testset "Group-level keyword constructors" begin
        v4_error = LinearVerbosity(error_control = SciMLLogging.Verbosity.Error())
        @test v4_error.default_lu_fallback isa SciMLLogging.Verbosity.Error

        v4_numerical = LinearVerbosity(numerical = SciMLLogging.Verbosity.Silent())
        @test v4_numerical.KrylovKit_verbosity isa SciMLLogging.Verbosity.Silent
        @test v4_numerical.using_IterativeSolvers isa SciMLLogging.Verbosity.Silent
        @test v4_numerical.pardiso_verbosity isa SciMLLogging.Verbosity.Silent

        v4_performance = LinearVerbosity(performance = SciMLLogging.Verbosity.Info())
        @test v4_performance.no_right_preconditioning isa SciMLLogging.Verbosity.Info
    end

    @testset "Mixed group and individual settings" begin
        v5_mixed = LinearVerbosity(
            numerical = SciMLLogging.Verbosity.Silent(),
            KrylovKit_verbosity = SciMLLogging.Verbosity.Warn(),
            performance = SciMLLogging.Verbosity.Info()
        )
        # Individual override should take precedence
        @test v5_mixed.KrylovKit_verbosity isa SciMLLogging.Verbosity.Warn
        # Other numerical options should use group setting
        @test v5_mixed.using_IterativeSolvers isa SciMLLogging.Verbosity.Silent
        # Performance group setting should apply
        @test v5_mixed.no_right_preconditioning isa SciMLLogging.Verbosity.Info
    end

    @testset "Individual keyword arguments" begin
        v6_individual = LinearVerbosity(
            default_lu_fallback = SciMLLogging.Verbosity.Error(),
            KrylovKit_verbosity = SciMLLogging.Verbosity.Info(),
            pardiso_verbosity = SciMLLogging.Verbosity.Silent()
        )
        @test v6_individual.default_lu_fallback isa SciMLLogging.Verbosity.Error
        @test v6_individual.KrylovKit_verbosity isa SciMLLogging.Verbosity.Info
        @test v6_individual.pardiso_verbosity isa SciMLLogging.Verbosity.Silent
        # Unspecified options should use defaults
        @test v6_individual.no_right_preconditioning isa SciMLLogging.Verbosity.Warn
    end

    @testset "Group classification functions" begin
        @test option_group(:default_lu_fallback) == :error_control
        @test option_group(:KrylovKit_verbosity) == :numerical
        @test option_group(:no_right_preconditioning) == :performance

        # Test error for unknown option
        @test_throws ErrorException option_group(:unknown_option)
    end

    @testset "Group options function" begin
        v8 = LinearVerbosity(numerical = SciMLLogging.Verbosity.Warn())
        numerical_opts = group_options(v8, :numerical)
        @test numerical_opts isa NamedTuple
        @test :KrylovKit_verbosity in keys(numerical_opts)
        @test :using_IterativeSolvers in keys(numerical_opts)
        @test numerical_opts.KrylovKit_verbosity isa SciMLLogging.Verbosity.Warn

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
        @test error_group.default_lu_fallback isa Verbosity.LogLevel
        @test performance_group.no_right_preconditioning isa Verbosity.LogLevel
        @test numerical_group.KrylovKit_verbosity isa Verbosity.LogLevel

        # Individual field access should still work
        @test v.default_lu_fallback isa Verbosity.Warn
        @test v.KrylovKit_verbosity isa Verbosity.Warn
    end

    @testset "Group setproperty! setting" begin
        v = LinearVerbosity()

        # Test setting entire error_control group
        v.error_control = Verbosity.Error()
        @test v.default_lu_fallback isa Verbosity.Error

        # Test setting entire performance group
        v.performance = Verbosity.Info()
        @test v.no_right_preconditioning isa Verbosity.Info

        # Test setting entire numerical group
        v.numerical = Verbosity.Silent()
        @test v.KrylovKit_verbosity isa Verbosity.Silent
        @test v.using_IterativeSolvers isa Verbosity.Silent
        @test v.pardiso_verbosity isa Verbosity.Silent
        @test v.HYPRE_verbosity isa Verbosity.Silent

        # Test that other groups aren't affected
        @test v.default_lu_fallback isa Verbosity.Error  # error_control unchanged
        @test v.no_right_preconditioning isa Verbosity.Info  # performance unchanged

        # Test individual setting still works after group setting
        v.KrylovKit_verbosity = Verbosity.Warn()
        @test v.KrylovKit_verbosity isa Verbosity.Warn
        # Other numerical options should still be Silent
        @test v.using_IterativeSolvers isa Verbosity.Silent
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
        v.numerical = Verbosity.Error()
        numerical_group = v.numerical

        @test all(x -> x isa Verbosity.Error, values(numerical_group))

        # Set individual option and verify both individual and group access work
        v.KrylovKit_verbosity = Verbosity.Info()
        @test v.KrylovKit_verbosity isa Verbosity.Info

        updated_numerical = v.numerical
        @test updated_numerical.KrylovKit_verbosity isa Verbosity.Info
        # Other numerical options should still be Error
        @test updated_numerical.using_IterativeSolvers isa Verbosity.Error
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
    verbose = LinearVerbosity(default_lu_fallback = Verbosity.Warn()))

@test_logs (:warn,
    "LU factorization failed, falling back to QR factorization. `A` is potentially rank-deficient.") solve(
    prob, verbose = true)

@test_logs min_level=SciMLLogging.Logging.Warn solve(prob, verbose = false)

@test_logs (:info,
    "LU factorization failed, falling back to QR factorization. `A` is potentially rank-deficient.") solve(
    prob,
    verbose = LinearVerbosity(default_lu_fallback = Verbosity.Info()))

verb = LinearVerbosity(default_lu_fallback = Verbosity.Warn())

@test_logs (:warn,
    "LU factorization failed, falling back to QR factorization. `A` is potentially rank-deficient.") solve(
    prob,
    verbose = verb)

verb.default_lu_fallback = Verbosity.Info()

@test_logs (:info,
    "LU factorization failed, falling back to QR factorization. `A` is potentially rank-deficient.") solve(
    prob,
    verbose = verb)