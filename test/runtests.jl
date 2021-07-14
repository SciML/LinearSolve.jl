using LinearSolve
using Test

@testset "LinearSolve.jl" begin
    A = rand(5, 5)
    b = rand(5)
    prob = LinearProblem(A, b)
    @test A * solve!(deepcopy(prob), LUFactorization()) ≈ b
    @test A * solve!(deepcopy(prob), QRFactorization()) ≈ b
    @test A * solve!(deepcopy(prob), SVDFactorization()) ≈ b
end
