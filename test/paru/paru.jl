using LinearSolve, LinearAlgebra, SparseArrays, Random, SciMLBase
using Test
import ParU_jll

Random.seed!(1234)

n = 8
A = Matrix(I, n, n)
b = ones(n)

A1 = sparse(A / 1)
b1 = rand(n)
x1 = zero(b)
A2 = sparse(A / 2)
b2 = rand(n)
x2 = zero(b)

prob1 = LinearProblem(A1, b1; u0 = x1)
prob2 = LinearProblem(A2, b2; u0 = x2)

cache_kwargs = (; verbose = true, abstol = 1e-8, reltol = 1e-8, maxiter = 30)

function test_interface(alg, prob1, prob2)
    A1, b1, x1 = prob1.A, prob1.b, prob1.u0
    A2, b2, x2 = prob2.A, prob2.b, prob2.u0

    y = solve(prob1, alg)
    @test A1 * y ≈ b1

    y = solve(prob2, alg)
    @test A2 * y ≈ b2
end

@testset "ParU Factorization" begin
    @test Base.get_extension(LinearSolve, :LinearSolveParUExt) !== nothing

    test_interface(ParUFactorization(), prob1, prob2)
    test_interface(ParUFactorization(reuse_symbolic = false), prob1, prob2)

    @testset "Cache reuse with matrix update" begin
        cache = SciMLBase.init(prob1, ParUFactorization(); cache_kwargs...)
        y = solve!(cache)
        @test A1 * y ≈ b1
        cache.A = A2
        @test A2 * solve!(cache) ≈ b1
    end
end
