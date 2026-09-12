using LinearSolve, LinearAlgebra, SparseArrays, Test

@testset "KLU refactorization refreshes unstable pivots" begin
    for T in (Float64, ComplexF64), Ti in unique((Int32, Int))
        A = SparseMatrixCSC{T, Ti}(sparse(T[1 1; 1 2]))
        expected = T[1, 2]
        cache = init(LinearProblem(A, A * expected), KLUFactorization())
        @test solve!(cache).u ≈ expected
        for pivot in (1.0e-10, 1.0e-16, 1.0e-20, 0.0, 1.0)
            A = SparseMatrixCSC{T, Ti}(sparse(T[1 1; 1 2]))
            A[1, 1] = pivot
            cache.A = A
            cache.b = A * expected
            solution = solve!(cache)
            @test SciMLBase.successful_retcode(solution)
            @test solution.u ≈ expected rtol = 1.0e-12 atol = 1.0e-12
            @test A * solution.u ≈ cache.b rtol = 1.0e-12 atol = 1.0e-12
        end
    end
end
