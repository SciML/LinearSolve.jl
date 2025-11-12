using LinearAlgebra, LinearSolve
using Test
using RecursiveFactorization

@testset "Random Matricies" begin
    for i in 400 : 500
        A = rand(i, i)
        b = rand(i)
        prob = LinearProblem(A, b)
        x = solve(prob, ButterflyFactorization())
        @test norm(A * x .- b) <= 1e-7
    end
end