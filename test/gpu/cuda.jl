using LinearSolve, CUDA, LinearAlgebra, SparseArrays
using Test

n = 8
A = Matrix(I, n, n)
b = ones(n)
A1 = A / 1;
b1 = rand(n);
x1 = zero(b);
A2 = A / 2;
b2 = rand(n);
x2 = zero(b);

prob1 = LinearProblem(A1, b1; u0 = x1)
prob2 = LinearProblem(A2, b2; u0 = x2)

cache_kwargs = (; verbose = true, abstol = 1e-8, reltol = 1e-8, maxiter = 30)

function test_interface(alg, prob1, prob2)
    A1 = prob1.A
    b1 = prob1.b
    x1 = prob1.u0
    A2 = prob2.A
    b2 = prob2.b
    x2 = prob2.u0

    y = solve(prob1, alg; cache_kwargs...)
    @test A1 * y ≈ b1

    cache = SciMLBase.init(prob1, alg; cache_kwargs...) # initialize cache
    y = solve(cache)
    @test A1 * y ≈ b1

    cache = LinearSolve.set_A(cache, copy(A2))
    y = solve(cache)
    @test A2 * y ≈ b1

    cache = LinearSolve.set_b(cache, b2)
    y = solve(cache)
    @test A2 * y ≈ b2

    return
end

@testset "CudaOffloadFactorization" begin
    test_interface(CudaOffloadFactorization(), prob1, prob2)
end

@testset "Simple GMRES: restart = $restart" for restart in (true, false)
    test_interface(SimpleGMRES(; restart), prob1, prob2)
end

A1 = prob1.A;
b1 = prob1.b;
x1 = prob1.u0;
y = solve(prob1)
@test A1 * y ≈ b1

using BlockDiagonals

@testset "Block Diagonal Specialization" begin
    A = BlockDiagonal([rand(2, 2) for _ in 1:3]) |> cu
    b = rand(size(A, 1)) |> cu

    x1 = zero(b)
    x2 = zero(b)
    prob1 = LinearProblem(A, b, x1)
    prob2 = LinearProblem(A, b, x2)

    test_interface(SimpleGMRES(; blocksize = 2), prob1, prob2)

    @test solve(prob1, SimpleGMRES(; blocksize = 2)).u ≈ solve(prob2, SimpleGMRES()).u
end
