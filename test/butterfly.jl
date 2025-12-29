using LinearAlgebra, LinearSolve
using Test
using RecursiveFactorization

@testset "Random Matrices" begin
    for i in 490 : 510
        A = rand(i, i)
        b = rand(i)
        prob = LinearProblem(A, b)
        x = solve(prob, ButterflyFactorization())
        @test norm(A * x .- b) <= 5e-6
    end
end

function wilkinson(N)
    A = zeros(N, N)
    A[1:(N+1):N*N] .= 1
    A[:, end] .= 1
    for n in 1:(N - 1)
        for r in (n + 1):N
            @inbounds A[r, n] = -1
        end
    end
    A
end

@testset "Wilkinson" begin
    for i in 790 : 810
        A = wilkinson(i)
        b = rand(i)
        prob = LinearProblem(A, b)
        x = solve(prob, ButterflyFactorization())
        @test norm(A * x .- b) <= 1e-9
    end
end
