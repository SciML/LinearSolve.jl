using LinearSolve
using Test

@testset "LinearSolve.jl" begin
    A = rand(5, 5)
    b = rand(5)
    prob = LinearProblem(A, b)
    @test A * solve(deepcopy(prob), LUFactorization();alias_A = false, alias_b = false) ≈ b
    @test A * solve(deepcopy(prob), QRFactorization();alias_A = false, alias_b = false) ≈ b
    @test A * solve(deepcopy(prob), SVDFactorization();alias_A = false, alias_b = false) ≈ b
end
