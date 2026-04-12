using LinearSolve, LinearAlgebra, SparseArrays, SciMLBase
using Test

@testset "STRUMPACK Factorization" begin
    ext = Base.get_extension(LinearSolve, :LinearSolveSTRUMPACKExt)
    @test ext !== nothing

    if ext === nothing || !ext.strumpack_isavailable()
        @test_throws ["STRUMPACKFactorization", "libstrumpack"] STRUMPACKFactorization()
        @test STRUMPACKFactorization(throwerror = false) isa STRUMPACKFactorization
        @test STRUMPACKFactorization(
            throwerror = false,
            options = ["--sp_compression", "hss", "--sp_rel_tol", "1e-4"]
        ) isa STRUMPACKFactorization

        alg_phase2 = STRUMPACKFactorization(
            throwerror = false,
            compression = "hss",
            rel_tol = 1.0e-4,
            abs_tol = 1.0e-10,
            max_rank = 64,
            leaf_size = 128,
            reordering = "metis",
            matching = true
        )
        @test alg_phase2.options == [
            "--sp_compression", "hss",
            "--sp_rel_tol", "0.0001",
            "--sp_abs_tol", "1.0e-10",
            "--sp_max_rank", "64",
            "--sp_leaf_size", "128",
            "--sp_reordering_method", "metis",
            "--sp_enable_matching", "1",
        ]

        @test_throws "`rel_tol` must be non-negative" STRUMPACKFactorization(
            throwerror = false,
            rel_tol = -1.0
        )
        @test_throws "`max_rank` must be >= 1" STRUMPACKFactorization(
            throwerror = false,
            max_rank = 0
        )

        A = sparse([4.0 1.0; 2.0 3.0])
        b = [1.0, -1.0]
        prob = LinearProblem(A, b)
        @test_throws ["STRUMPACKFactorization", "libstrumpack"] solve(
            prob,
            STRUMPACKFactorization(throwerror = false)
        )
    else
        A = sparse(
            [
                7.0 1.0 0.0
                2.0 8.0 1.0
                0.0 3.0 9.0
            ]
        )
        b = [1.0, -2.0, 3.0]

        prob = LinearProblem(A, b)
        sol = solve(prob, STRUMPACKFactorization())
        @test sol.retcode == ReturnCode.Success
        @test A * sol.u ≈ b atol = 1.0e-10 rtol = 1.0e-10

        alg_opts = STRUMPACKFactorization(
            options = ["--sp_compression", "hss", "--sp_rel_tol", "1e-4"]
        )
        @test alg_opts.options == ["--sp_compression", "hss", "--sp_rel_tol", "1e-4"]

        alg_phase2 = STRUMPACKFactorization(
            compression = "hss",
            rel_tol = 1.0e-4,
            abs_tol = 1.0e-10,
            max_rank = 64,
            leaf_size = 128,
            reordering = "metis",
            matching = true
        )
        @test alg_phase2.options == [
            "--sp_compression", "hss",
            "--sp_rel_tol", "0.0001",
            "--sp_abs_tol", "1.0e-10",
            "--sp_max_rank", "64",
            "--sp_leaf_size", "128",
            "--sp_reordering_method", "metis",
            "--sp_enable_matching", "1",
        ]

        @test_throws "`rel_tol` must be non-negative" STRUMPACKFactorization(
            rel_tol = -1.0
        )
        @test_throws "`max_rank` must be >= 1" STRUMPACKFactorization(
            max_rank = 0
        )

        cache = init(prob, STRUMPACKFactorization())
        sol1 = solve!(cache)
        @test sol1.retcode == ReturnCode.Success
        @test A * sol1.u ≈ b atol = 1.0e-10 rtol = 1.0e-10

        A2 = sparse(
            [
                8.0 1.0 0.0
                2.0 9.0 1.0
                0.0 3.0 10.0
            ]
        )
        cache.A = A2
        sol2 = solve!(cache)
        @test sol2.retcode == ReturnCode.Success
        @test A2 * sol2.u ≈ b atol = 1.0e-10 rtol = 1.0e-10

        prob_guess = LinearProblem(A, b; u0 = fill(1.0, length(b)))
        sol_guess = solve(prob_guess, STRUMPACKFactorization(use_initial_guess = true))
        @test sol_guess.retcode == ReturnCode.Success
        @test A * sol_guess.u ≈ b atol = 1.0e-10 rtol = 1.0e-10
    end
end
