using LinearSolve, LinearAlgebra, SparseArrays, Test

@testset "PureKLU refactorization refreshes unstable pivots" begin
    algorithms = (
        PureKLUFactorization(),
        LinearSolve.DefaultLinearSolver(LinearSolve.DefaultAlgorithmChoice.KLUFactorization),
    )
    for T in (Float64, ComplexF64), Ti in unique((Int32, Int)), algorithm in algorithms
        A = SparseMatrixCSC{T, Ti}(sparse(T[1 1; 1 2]))
        expected = T[1, 2]
        cache = init(LinearProblem(A, A * expected), algorithm)
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

@testset "PureKLU refactorization keeps pivots that are still stable" begin
    # Refactorizing with fresh pivots costs a full `klu_factor!` where reusing
    # them costs a `klu_refactor!` — ~2.4x on a banded operand — so the pivots
    # are refreshed only when the growth check says they have gone unstable.
    # `1e-4` on the diagonal is below KLU's `tol` (so a full factorization
    # would pivot onto the other row) while leaving growth far above
    # `sqrt(eps(Float64))`: the pivot order must survive.
    A = sparse([1.0 1.0; 1.0 3.0])
    expected = [1.0, 2.0]
    cache = init(LinearProblem(A, A * expected), PureKLUFactorization())
    @test solve!(cache).u ≈ expected
    pivots = copy(cache.cacheval.p)

    B = sparse([1.0e-4 1.0; 1.0 3.0])
    cache.A = B
    cache.b = B * expected
    solution = solve!(cache)
    @test SciMLBase.successful_retcode(solution)
    @test solution.u ≈ expected rtol = 1.0e-10
    @test cache.cacheval.p == pivots
end
