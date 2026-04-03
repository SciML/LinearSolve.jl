using LinearSolve, LinearAlgebra, SparseArrays, SciMLBase
using Test

@testset "STRUMPACK Factorization" begin
    ext = Base.get_extension(LinearSolve, :LinearSolveSTRUMPACKExt)
    @test ext !== nothing

    if ext === nothing || !ext.strumpack_isavailable()
        @test_throws ["STRUMPACKFactorization", "libstrumpack"] STRUMPACKFactorization()
        @test STRUMPACKFactorization(throwerror = false) isa STRUMPACKFactorization

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
