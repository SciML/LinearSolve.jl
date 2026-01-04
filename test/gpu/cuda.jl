using LinearSolve, CUDA, LinearAlgebra, SparseArrays, StableRNGs
using CUDA.CUSPARSE
using Test

@testset "Test default solver choice for CuSparse" begin
    b = Float64[1, 2, 3, 4]
    b_gpu = CUDA.adapt(CuArray, b)

    A = Float64[
        1 1 0 0
        0 1 1 0
        0 0 3 1
        0 0 0 4
    ]
    A_gpu_csr = CUDA.CUSPARSE.CuSparseMatrixCSR(sparse(A))
    A_gpu_csc = CUDA.CUSPARSE.CuSparseMatrixCSC(sparse(A))
    prob_csr = LinearProblem(A_gpu_csr, b_gpu)
    prob_csc = LinearProblem(A_gpu_csc, b_gpu)

    A_sym = Float64[
        1 1 0 0
        1 0 0 2
        0 0 3 0
        0 2 0 0
    ]
    A_gpu_sym_csr = CUDA.CUSPARSE.CuSparseMatrixCSR(sparse(A_sym))
    A_gpu_sym_csc = CUDA.CUSPARSE.CuSparseMatrixCSC(sparse(A_sym))
    prob_sym_csr = LinearProblem(A_gpu_sym_csr, b_gpu)
    prob_sym_csc = LinearProblem(A_gpu_sym_csc, b_gpu)

    @testset "Test without CUDSS loaded" begin
        # assert CuDSS is not loaded yet
        @test !LinearSolve.cudss_loaded(A_gpu_csr)
        # csr fallback to krylov
        alg = solve(prob_csr).alg
        @test alg.alg == LinearSolve.DefaultAlgorithmChoice.KrylovJL_GMRES
        # csc fallback to krylov
        alg = solve(prob_csc).alg
        @test alg.alg == LinearSolve.DefaultAlgorithmChoice.KrylovJL_GMRES
        # csr symmetric fallback to krylov
        alg = solve(prob_sym_csr).alg
        @test alg.alg == LinearSolve.DefaultAlgorithmChoice.KrylovJL_GMRES
        # csc symmetric fallback to krylov
        alg = solve(prob_sym_csc).alg
        @test alg.alg == LinearSolve.DefaultAlgorithmChoice.KrylovJL_GMRES
    end

    using CUDSS

    @testset "Test with CUDSS loaded" begin
        @test LinearSolve.cudss_loaded(A_gpu_csr)
        # csr uses LU
        alg = solve(prob_csr).alg
        @test alg.alg == LinearSolve.DefaultAlgorithmChoice.LUFactorization
        # csc fallback to krylov
        alg = solve(prob_csc).alg
        @test alg.alg == LinearSolve.DefaultAlgorithmChoice.KrylovJL_GMRES
        # csr symmetric uses LU/cholesky
        alg = solve(prob_sym_csr).alg
        @test alg.alg == LinearSolve.DefaultAlgorithmChoice.LUFactorization
        # csc symmetric fallback to krylov
        alg = solve(prob_sym_csc).alg
        @test alg.alg == LinearSolve.DefaultAlgorithmChoice.KrylovJL_GMRES
    end
end

CUDA.allowscalar(false)

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

cache_kwargs = (; abstol = 1.0e-8, reltol = 1.0e-8, maxiter = 30)

function test_interface(alg, prob1, prob2)
    A1 = prob1.A
    b1 = prob1.b
    x1 = prob1.u0
    A2 = prob2.A
    b2 = prob2.b
    x2 = prob2.u0

    y = solve(prob1, alg; cache_kwargs...)
    @test CUDA.@allowscalar(Array(A1 * y) ≈ Array(b1))

    cache = SciMLBase.init(prob1, alg; cache_kwargs...) # initialize cache
    solve!(cache)
    @test CUDA.@allowscalar(Array(A1 * cache.u) ≈ Array(b1))

    cache.A = copy(A2)
    solve!(cache)
    @test CUDA.@allowscalar(Array(A2 * cache.u) ≈ Array(b1))

    cache.b = copy(b2)
    solve!(cache)
    @test CUDA.@allowscalar(Array(A2 * cache.u) ≈ Array(b2))

    return
end

@testset "$alg" for alg in (CudaOffloadLUFactorization(), CudaOffloadQRFactorization(), NormalCholeskyFactorization())
    test_interface(alg, prob1, prob2)
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

    x1 = zero(b) |> cu
    x2 = zero(b) |> cu
    prob1 = LinearProblem(A, b, x1)
    prob2 = LinearProblem(A, b, x2)

    test_interface(SimpleGMRES(; blocksize = 2), prob1, prob2)

    @test solve(prob1, SimpleGMRES(; blocksize = 2)).u ≈ solve(prob2, SimpleGMRES()).u
end

# Test Dispatches for Adjoint/Transpose Types
rng = StableRNG(0)

A = Matrix(Hermitian(rand(rng, 5, 5) + I)) |> cu
b = rand(rng, 5) |> cu
prob1 = LinearProblem(A', b)
prob2 = LinearProblem(transpose(A), b)

@testset "Adjoint/Transpose Type: $(alg)" for alg in (
        NormalCholeskyFactorization(),
        CholeskyFactorization(), LUFactorization(), QRFactorization(), nothing,
    )
    sol = solve(
        prob1, alg;
        alias = LinearAliasSpecifier(alias_A = false)
    )
    @test norm(A' * sol.u .- b) < 1.0e-5

    sol = solve(prob2, alg; alias = LinearAliasSpecifier(alias_A = false))
    @test norm(transpose(A) * sol.u .- b) < 1.0e-5
end

@testset "CUDSS" begin
    T = Float32
    n = 100
    A_cpu = sprand(rng, T, n, n, 0.05) + I
    x_cpu = zeros(T, n)
    b_cpu = rand(rng, T, n)

    A_gpu_csr = CuSparseMatrixCSR(A_cpu)
    b_gpu = CuVector(b_cpu)

    prob = LinearProblem(A_gpu_csr, b_gpu)
    sol = solve(prob)
end

# Include CUSOLVERRF tests if available
if Base.find_package("CUSOLVERRF") !== nothing
    @testset "CUSOLVERRF" begin
        include("cusolverrf.jl")
    end
end
