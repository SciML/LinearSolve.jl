using LinearSolve, LinearAlgebra, SparseArrays, StableRNGs
using CUDACore, cuSPARSE, cuSOLVER # or using CUDA, cuSPARSE
using Test

@testset "Test default solver choice for CuSparse" begin
    b = Float64[1, 2, 3, 4]
    b_gpu = CUDACore.adapt(CuArray, b)

    A = Float64[
        1 1 0 0
        0 1 1 0
        0 0 3 1
        0 0 0 4
    ]
    A_gpu_csr = cuSPARSE.CuSparseMatrixCSR(sparse(A))
    A_gpu_csc = cuSPARSE.CuSparseMatrixCSC(sparse(A))
    prob_csr = LinearProblem(A_gpu_csr, b_gpu)
    prob_csc = LinearProblem(A_gpu_csc, b_gpu)

    A_sym = Float64[
        1 1 0 0
        1 0 0 2
        0 0 3 0
        0 2 0 0
    ]
    A_gpu_sym_csr = cuSPARSE.CuSparseMatrixCSR(sparse(A_sym))
    A_gpu_sym_csc = cuSPARSE.CuSparseMatrixCSC(sparse(A_sym))
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

CUDACore.allowscalar(false)

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
    @test CUDACore.@allowscalar(Array(A1 * y) ≈ Array(b1))

    cache = SciMLBase.init(prob1, alg; cache_kwargs...) # initialize cache
    solve!(cache)
    @test CUDACore.@allowscalar(Array(A1 * cache.u) ≈ Array(b1))

    cache.A = copy(A2)
    solve!(cache)
    @test CUDACore.@allowscalar(Array(A2 * cache.u) ≈ Array(b1))

    cache.b = copy(b2)
    solve!(cache)
    @test CUDACore.@allowscalar(Array(A2 * cache.u) ≈ Array(b2))

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
    # Seeded, Float32 end-to-end, and gated against the direct solution. The
    # earlier form compared the blocked and unblocked iterative answers to each
    # other at the default isapprox rtol (~3.4f-4): on GPU each path converges
    # to within the solver's own Float32 tolerance, so two correct answers can
    # legitimately differ by ~1f-3 and the comparison flaked (measured 9/400
    # draws on a T4; worst observed error vs direct was 8.6f-4).
    rng_bd = StableRNG(42)
    A_cpu = BlockDiagonal([rand(rng_bd, Float32, 2, 2) + 2.0f0 * I for _ in 1:3])
    b_cpu = rand(rng_bd, Float32, size(A_cpu, 1))
    ref = Matrix(A_cpu) \ b_cpu
    A = A_cpu |> cu
    b = b_cpu |> cu

    x1 = zero(b) |> cu
    x2 = zero(b) |> cu
    prob1 = LinearProblem(A, b, x1)
    prob2 = LinearProblem(A, b, x2)

    test_interface(SimpleGMRES(; blocksize = 2), prob1, prob2)

    u1 = solve(prob1, SimpleGMRES(; blocksize = 2)).u
    u2 = solve(prob2, SimpleGMRES()).u
    @test Array(u1) ≈ ref rtol = 5.0f-3
    @test Array(u2) ≈ ref rtol = 5.0f-3
    @test u1 ≈ u2 rtol = 5.0f-3
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

# A `WOperator` with a device Jacobian used to default to `LHLFactorization`, whose
# reduction runs on the host: `CuArray <: DenseArray`, so `A.J isa DenseMatrix` holds on
# the GPU and `_lhl_defaultable` accepted it. The solve then died on scalar indexing.
@testset "WOperator on the GPU does not default to LHLFactorization" begin
    n = 64
    γ = 0.1
    J = CUDACore.adapt(CuArray, rand(n, n) + n * I)
    b_gpu = CUDACore.adapt(CuArray, rand(n))
    W = LinearSolve.WOperator{true}(I, γ, J, similar(b_gpu))

    @test LinearSolve.defaultalg(W, b_gpu, OperatorAssumptions(true)).alg !==
        LinearSolve.DefaultAlgorithmChoice.LHLFactorization

    sol = solve(LinearProblem(W, b_gpu))
    @test SciMLBase.successful_retcode(sol)
    ref = Array(J - I / γ) \ Array(b_gpu)
    @test Array(sol.u) ≈ ref rtol = 1.0e-6

    # A host `WOperator` of the same shape still takes the LHL path.
    Jc = rand(n, n) + n * I
    bc = rand(n)
    Wc = LinearSolve.WOperator{true}(I, γ, Jc, similar(bc))
    @test LinearSolve.defaultalg(Wc, bc, OperatorAssumptions(true)).alg ===
        LinearSolve.DefaultAlgorithmChoice.LHLFactorization
end

# Two separate failures kept a non-square GPU `A` from solving at all.
#
# `_init_default_cacheval` builds a cacheval for every algorithm slot before it knows
# which one it will use, and two of those slots called `cholesky` on `A` itself, so
# `init` threw `DimensionMismatch` before any algorithm ran. Then for a wide `A`, the
# QR solve itself threw: a GPU `qr` factors a wide matrix, but solving with the result
# builds `UpperTriangular(R)` on a non-square `R`.
# See https://github.com/SciML/NonlinearSolve.jl/issues/746 and #857.
@testset "Non-square GPU matrices" begin
    tall = CUDACore.adapt(CuArray, Float32[1 2; 3 4; 5 6; 7 8])
    btall = CUDACore.adapt(CuArray, Float32[1, 2, 3, 4])
    wide = CUDACore.adapt(CuArray, Float32[1 2 3 4; 5 6 7 8])
    bwide = CUDACore.adapt(CuArray, Float32[1, 2])

    @testset "$name" for (name, A, b) in (("tall", tall, btall), ("wide", wide, bwide))
        ref = Array(A) \ Array(b)

        # `init` is where every slot gets built, and where this used to throw.
        cache = init(LinearProblem(A, b))
        sol = solve!(cache)
        @test SciMLBase.successful_retcode(sol)
        @test Array(sol.u) ≈ ref rtol = 1.0e-4

        @test LinearSolve.defaultalg(A, b, OperatorAssumptions(false)).alg ===
            LinearSolve.DefaultAlgorithmChoice.QRFactorization
        @test Array(solve(LinearProblem(A, b), QRFactorization()).u) ≈ ref rtol = 1.0e-4
    end

    # The wide solve has to be the minimum-norm one, which is what dense `\` gives on
    # the CPU. Agreeing on the residual alone would not distinguish it from any other
    # point on the solution manifold.
    xwide = Array(solve(LinearProblem(wide, bwide)).u)
    @test norm(xwide) ≈ norm(Array(wide) \ Array(bwide)) rtol = 1.0e-4
    @test Array(wide) * xwide ≈ Array(bwide) rtol = 1.0e-4
end
