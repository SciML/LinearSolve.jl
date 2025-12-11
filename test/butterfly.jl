using LinearSolve, LinearAlgebra, Test
using StableRNGs

@testset "Butterfly Factorization" begin
    @testset "Random Matrices" begin
        rng = StableRNG(12345)
        for n in 2:2:42
            A = randn(rng, n, n)
            b = randn(rng, n)
            prob = LinearProblem(A, b)
            x = solve(prob).u
            @test norm(A * x .- b) <= 1.0e-5
        end
    end

    @testset "Wilkinson" begin
        for n in 2:2:42
            # Wilkinson matrix: tridiagonal with specific structure
            A = diagm(-1 => ones(n - 1), 0 => abs.(collect(1:n) .- (n + 1) / 2), 1 => ones(n - 1))
            b = ones(n)
            prob = LinearProblem(A, b)
            x = solve(prob).u
            @test norm(A * x .- b) <= 1.0e-10
        end
    end
end
