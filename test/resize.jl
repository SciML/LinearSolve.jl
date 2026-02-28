using LinearSolve, LinearAlgebra, Test

@testset "LinearCache resize!" begin
    # Helper: create a cache with a given algorithm, resize it, and solve
    function test_resize(
            alg; n_init = 3, n_new = 6, atol = 1.0e-10,
            check_retcode = true, kwargs...
        )
        A_init = rand(n_init, n_init) + 5I
        b_init = rand(n_init)
        prob = LinearProblem(A_init, b_init)
        cache = init(
            prob, alg;
            alias = LinearAliasSpecifier(alias_A = false, alias_b = false),
            kwargs...
        )

        # Solve at original size to populate cacheval
        sol1 = solve!(cache)
        if check_retcode
            @test sol1.retcode == ReturnCode.Success
        end
        @test length(sol1.u) == n_init

        # Resize
        resize!(cache, n_new)
        @test cache.isfresh == true

        # Set new A, b, u of the new size
        A_new = rand(n_new, n_new) + 5I
        b_new = rand(n_new)
        u_new = zeros(n_new)

        # Compute expected solution BEFORE solve (which may modify A in-place)
        expected = A_new \ b_new

        cache.A = A_new
        cache.b = b_new
        cache.u = u_new

        # Solve at new size
        sol2 = solve!(cache)
        if check_retcode
            @test sol2.retcode == ReturnCode.Success
        end
        @test length(sol2.u) == n_new
        @test sol2.u ≈ expected atol = atol
    end

    @testset "Default (nothing)" begin
        test_resize(nothing)
    end

    @testset "LUFactorization" begin
        test_resize(LUFactorization())
    end

    @testset "QRFactorization" begin
        test_resize(QRFactorization())
    end

    @testset "SVDFactorization" begin
        test_resize(SVDFactorization())
    end

    @testset "GenericLUFactorization" begin
        test_resize(GenericLUFactorization())
    end

    @testset "KrylovJL_GMRES" begin
        test_resize(KrylovJL_GMRES(); atol = 1.0e-6, maxiters = 100)
    end

    @testset "SimpleLUFactorization" begin
        test_resize(SimpleLUFactorization(); check_retcode = false)
    end

    @testset "NormalCholeskyFactorization" begin
        test_resize(NormalCholeskyFactorization())
    end

    @testset "CholeskyFactorization" begin
        # Cholesky requires SPD matrix — custom test
        A_init = let X = rand(3, 3)
            X' * X + 5I
        end
        prob = LinearProblem(A_init, rand(3))
        cache = init(
            prob, CholeskyFactorization();
            alias = LinearAliasSpecifier(alias_A = false, alias_b = false)
        )
        solve!(cache)

        resize!(cache, 6)
        @test cache.isfresh == true

        A_new = let X = rand(6, 6)
            X' * X + 5I
        end
        b_new = rand(6)
        expected = A_new \ b_new
        cache.A = A_new
        cache.b = b_new
        cache.u = zeros(6)

        sol = solve!(cache)
        @test sol.retcode == ReturnCode.Success
        @test length(sol.u) == 6
        @test sol.u ≈ expected
    end

    @testset "SimpleGMRES" begin
        test_resize(SimpleGMRES(); atol = 1.0e-6, maxiters = 100)
    end

    @testset "DefaultLinearSolverInit A_backup resize" begin
        A = rand(3, 3) + 5I
        b = rand(3)
        prob = LinearProblem(A, b)
        cache = init(prob, nothing)
        @test cache.cacheval isa LinearSolve.DefaultLinearSolverInit
        @test size(cache.cacheval.A_backup) == (3, 3)

        resize!(cache, 5)
        @test size(cache.cacheval.A_backup) == (5, 5)
        @test cache.isfresh == true
    end

    @testset "resize then setproperty! with same object" begin
        # Reproduces the exact OrdinaryDiffEq crash scenario
        A = rand(3, 3) + 5I
        b = rand(3)
        prob = LinearProblem(A, b)
        cache = init(
            prob, nothing;
            alias = LinearAliasSpecifier(alias_A = false, alias_b = false)
        )

        sol = solve!(cache)
        @test sol.retcode == ReturnCode.Success

        # Set A to a new larger matrix
        A_new = rand(5, 5) + 5I
        b_new = rand(5)
        cache.A = A_new
        cache.b = b_new
        cache.u = zeros(5)
        sol = solve!(cache)
        @test sol.retcode == ReturnCode.Success

        # Set A to the same object again (in-place mutation + re-assignment)
        A_new .= rand(5, 5) + 5I
        b_new .= rand(5)
        expected = A_new \ b_new
        @test_nowarn (cache.A = A_new)
        cache.b = b_new

        sol = solve!(cache)
        @test sol.retcode == ReturnCode.Success
        @test sol.u ≈ expected
    end

    @testset "multiple resizes" begin
        A = rand(3, 3) + 5I
        b = rand(3)
        prob = LinearProblem(A, b)
        cache = init(
            prob, LUFactorization();
            alias = LinearAliasSpecifier(alias_A = false, alias_b = false)
        )

        sol = solve!(cache)
        @test sol.retcode == ReturnCode.Success

        # Resize up
        resize!(cache, 7)
        A2 = rand(7, 7) + 5I
        b2 = rand(7)
        expected2 = A2 \ b2
        cache.A = A2
        cache.b = b2
        cache.u = zeros(7)
        sol = solve!(cache)
        @test sol.retcode == ReturnCode.Success
        @test sol.u ≈ expected2

        # Resize down
        resize!(cache, 2)
        A3 = rand(2, 2) + 5I
        b3 = rand(2)
        expected3 = A3 \ b3
        cache.A = A3
        cache.b = b3
        cache.u = zeros(2)
        sol = solve!(cache)
        @test sol.retcode == ReturnCode.Success
        @test sol.u ≈ expected3
    end
end
