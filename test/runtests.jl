using LinearSolvers
using Test

@testset "LinearSolvers.jl" begin
    A = rand(5, 5)
    b = rand(5)
    prob = LinearProblem(A, b)
    @test A * solve(prob, LUFactorization(Val(true))) ≈ b
end
