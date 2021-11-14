using LinearSolve
using Test

@testset "LinearSolve.jl" begin
    n = 10
    A = rand(n, n)
    b = rand(n)
    prob = LinearProblem(A, b)

    x = rand(n)

    # Factorization
    @test A * solve(prob, LUFactorization();)  ≈ b
    @test A * solve(prob, QRFactorization();)  ≈ b
    @test A * solve(prob, SVDFactorization();) ≈ b

    # Krylov
    @test A * solve(prob, KrylovJL(A, b)) ≈ b

    # make algorithm callable - interoperable with DiffEq ecosystem
    @test A * LUFactorization()(x,A,b) ≈ b 
    @test_broken A * x ≈ b # in place
end
