using LinearSolve
using Test

@testset "LinearSolve.jl" begin
    A = rand(5, 5)
    b = rand(5)
    prob = LinearProblem(A, b)

    # Factorization
    @test A * solve(prob, LUFactorization(); alias_A = false, alias_b = false) ≈
          b
    @test A * solve(prob, QRFactorization(); alias_A = false, alias_b = false) ≈
          b
    @test A *
          solve(prob, SVDFactorization(); alias_A = false, alias_b = false) ≈ b

    # Krylov
    @test A * solve(prob, KrylovJL(A, b)) ≈ b
end
