using LinearSolve, LinearAlgebra, Test

@testset "SLATEFactorization" begin
    alg = SLATEFactorization()
    A = [3.0 1.0; 1.0 2.0]
    b = [1.0, 4.0]
    prob = LinearProblem(A, b)

    if slate_isavailable()
        sol = solve(prob, alg)
        @test SciMLBase.successful_retcode(sol.retcode)
        @test A * sol.u ≈ b

        B = [1.0 2.0; 4.0 5.0]
        matprob = LinearProblem(A, B)
        matsol = solve(matprob, alg)
        @test SciMLBase.successful_retcode(matsol.retcode)
        @test A * matsol.u ≈ B
    else
        @test_throws ErrorException solve(prob, alg)
    end
end
