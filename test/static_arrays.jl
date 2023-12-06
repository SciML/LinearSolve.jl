using LinearSolve, StaticArrays, LinearAlgebra

A = SMatrix{5, 5}(Hermitian(rand(5, 5) + I))
b = SVector{5}(rand(5))

for alg in (nothing, LUFactorization(), SVDFactorization(), CholeskyFactorization(),
    KrylovJL_GMRES())
    sol = solve(LinearProblem(A, b), alg)
    @test norm(A * sol .- b) < 1e-10
end

A = SMatrix{7, 5}(rand(7, 5))
b = SVector{7}(rand(7))

for alg in (nothing, SVDFactorization(), KrylovJL_LSMR())
    @test_nowarn solve(LinearProblem(A, b), alg)
end

A = SMatrix{5, 7}(rand(5, 7))
b = SVector{5}(rand(5))

for alg in (nothing, SVDFactorization(), KrylovJL_LSMR())
    @test_nowarn solve(LinearProblem(A, b), alg)
end
