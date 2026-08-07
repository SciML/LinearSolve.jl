using LinearSolve, LinearAlgebra, Random, SciMLBase, Test

# Sizes deliberately straddle the cutoff so both the scalar kernel and the BLAS
# fallback stay exercised rather than one silently rotting.
const SIZES = [1, 2, 3, 8, 63, 64, 65, 96]

# Unstructured Gaussian matrices, so partial pivoting actually permutes rows. A
# diagonally dominant test matrix would leave `ipiv == 1:n` and silently accept a
# back-solve that skipped the interchanges entirely.
testmat(rng, T, n) = randn(rng, T, n, n)

backward_error(A, x, b) = norm(A * x - b) / (opnorm(A, 1) * norm(x) + norm(b))

@testset "sizes straddle DENSE_BLAS_CUTOFF" begin
    @test any(<=(LinearSolve.DENSE_BLAS_CUTOFF), SIZES)
    @test any(>(LinearSolve.DENSE_BLAS_CUTOFF), SIZES)
end

# At n >= 8 an unstructured Gaussian matrix leaving every pivot on the diagonal
# is vanishingly unlikely, so this pins the property without being flaky the way
# n = 2 would be.
@testset "test matrices exercise the row interchanges" begin
    rng = MersenneTwister(0x5ca1ab1e)
    big = filter(>=(8), SIZES)
    pivoted = count(big) do n
        F = lu!(testmat(rng, Float64, n), check = false)
        any(i -> F.ipiv[i] != i, eachindex(F.ipiv))
    end
    @test pivoted == length(big)
end

@testset "_ldiv! matches LinearAlgebra for $T" for T in (Float64, Float32, ComplexF64)
    rng = MersenneTwister(0xf00d)
    tol = real(T) === Float32 ? 1.0f-4 : 1.0e-12
    for n in SIZES
        A = testmat(rng, T, n)
        b = randn(rng, T, n)
        F = lu!(copy(A), check = false)
        expected = ldiv!(similar(b), F, b)

        got = similar(b)
        LinearSolve._ldiv!(got, F, b)
        @test backward_error(A, got, b) <= tol
        @test norm(got - expected) <= tol * max(norm(expected), one(real(T)))

        # in-place aliasing: x === b must still solve rather than corrupt
        aliased = copy(b)
        LinearSolve._ldiv!(aliased, F, aliased)
        @test backward_error(A, aliased, b) <= tol
    end
end

# A pivot-forcing matrix: without the row interchange the first elimination
# divides by 1e-14 and the result is garbage.
@testset "matrix requiring an interchange" begin
    A = [1.0e-14 1.0; 1.0 1.0]
    b = [1.0, 2.0]
    F = lu!(copy(A), check = false)
    @test F.ipiv != [1, 2]
    got = similar(b)
    LinearSolve._ldiv!(got, F, b)
    @test backward_error(A, got, b) <= 1.0e-12
end

@testset "$(nameof(typeof(alg))) solves correctly across the cutoff" for alg in (
        GenericLUFactorization(), LUFactorization(),
    )
    rng = MersenneTwister(0xbeef)
    for n in SIZES
        A = testmat(rng, Float64, n)
        b = randn(rng, n)
        sol = solve(LinearProblem(copy(A), copy(b)), alg)
        @test SciMLBase.successful_retcode(sol)
        @test backward_error(A, sol.u, b) <= 1.0e-12
    end
end

@testset "default algorithm solves correctly across the cutoff" begin
    rng = MersenneTwister(0xcafe)
    for n in SIZES
        A = testmat(rng, Float64, n)
        b = randn(rng, n)
        sol = solve(LinearProblem(copy(A), copy(b)))
        @test backward_error(A, sol.u, b) <= 1.0e-12
    end
end

# A singular matrix must still be reported through the factorization's own
# success check, not silently divided by a zero pivot in the scalar kernel.
@testset "singular input is reported" begin
    sol = solve(LinearProblem(zeros(4, 4), randn(4)), GenericLUFactorization())
    @test !SciMLBase.successful_retcode(sol)
end
