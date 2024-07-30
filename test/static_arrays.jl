using LinearSolve, StaticArrays, LinearAlgebra, Test, StableRNGs
using AllocCheck

rng = StableRNG(0)

A = SMatrix{5, 5}(Hermitian(rand(rng, 5, 5) + I))
b = SVector{5}(rand(rng, 5))

@check_allocs __solve_no_alloc(A, b, alg) = solve(LinearProblem(A, b), alg)

function __non_native_static_array_alg(alg)
    return alg isa SVDFactorization || alg isa KrylovJL
end

for alg in (nothing, LUFactorization(), SVDFactorization(), CholeskyFactorization(),
    NormalCholeskyFactorization(), KrylovJL_GMRES())
    sol = solve(LinearProblem(A, b), alg)
    @inferred solve(LinearProblem(A, b), alg)
    @test norm(A * sol .- b) < 1e-10
    if alg isa KrylovJL{typeof(LinearSolve.Krylov.gmres!)}
        @test_broken __solve_no_alloc(A, b, alg) isa SciMLBase.LinearSolution
    else
        @test_nowarn __solve_no_alloc(A, b, alg) isa SciMLBase.LinearSolution
    end
    cache = init(LinearProblem(A, b), alg)
    sol = solve!(cache)
    @test norm(A * sol .- b) < 1e-10
end

A = SMatrix{7, 5}(rand(rng, 7, 5))
b = SVector{7}(rand(rng, 7))

for alg in (nothing, SVDFactorization(), KrylovJL_LSMR())
    @inferred solve(LinearProblem(A, b), alg)
    @test_nowarn solve(LinearProblem(A, b), alg)
end

A = SMatrix{5, 7}(rand(rng, 5, 7))
b = SVector{5}(rand(rng, 5))

for alg in (nothing, SVDFactorization(), KrylovJL_LSMR())
    @inferred solve(LinearProblem(A, b), alg)
    @test_nowarn solve(LinearProblem(A, b), alg)
end
