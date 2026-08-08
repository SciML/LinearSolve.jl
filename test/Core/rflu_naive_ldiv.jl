using LinearSolve, LinearAlgebra, RecursiveFactorization, Test, Random

Random.seed!(1161)

const RFExt = Base.get_extension(LinearSolve, :LinearSolveRecursiveFactorizationExt)
const CUTOFF = RFExt.NAIVE_LDIV_MAXSIZE

# A row-reversed diagonally dominant matrix: well conditioned, but partial
# pivoting has to undo the reversal, so a back-solve that skips the row
# interchanges gets a wrong answer rather than the same answer.
pivoting_matrix(T, n) = (rand(T, n, n) + n * I)[n:-1:1, :]

rf_lu(A) = RecursiveFactorization.lu!(
    copy(A), Vector{LinearAlgebra.BlasInt}(undef, min(size(A)...)),
    Val(true), Val(true), check = false
)

@testset "RFLU back-solve correctness across the naive-kernel cutoff" begin
    for T in (Float64, Float32, ComplexF64, ComplexF32)
        @testset "$T" begin
            rtol = 100 * eps(float(real(one(T))))
            sizes = T === Float64 ? (2, 5, 20, 64, CUTOFF, CUTOFF + 1) : (2, 5, 20, 64)
            for n in sizes
                A = pivoting_matrix(T, n)
                # The pivot-sensitivity of this testset rests on interchanges
                # actually firing.
                @test rf_lu(A).ipiv != collect(1:n)
                for rhs in (rand(T, n), rand(T, n, 1), rand(T, n, 4))
                    sol = solve(LinearProblem(copy(A), copy(rhs)), RFLUFactorization())
                    @test SciMLBase.successful_retcode(sol)
                    @test sol.u ≈ A \ rhs rtol = rtol * n
                end
            end
        end
    end
end

# Discriminator for #1161: at or below the cutoff the single-RHS path must
# produce exactly what the naive kernel produces, bit for bit. `getrs!` solves
# the same system to a different rounding, so this fails if the path reverts to
# `ldiv!`.
@testset "RFLU single-RHS back-solve runs the naive kernel" begin
    for n in (2, 5, 20, 64, 128, CUTOFF)
        A = pivoting_matrix(Float64, n)
        b = rand(n)
        fact = rf_lu(A)
        @test fact.ipiv != collect(1:n)

        ref = copy(b)
        LinearSolve._naive_lu_ldiv!(fact.factors, fact.ipiv, ref)
        @test ref ≈ A \ b rtol = 1.0e-10

        u = similar(b)
        RFExt._rf_ldiv!(u, fact, b, Val(true))
        @test u == ref

        U = similar(b, n, 1)
        RFExt._rf_ldiv!(U, fact, reshape(b, n, 1), Val(true))
        @test vec(U) == ref

        # In-place aliasing (`u === b`) must solve rather than zero the RHS.
        aliased = copy(b)
        RFExt._rf_ldiv!(aliased, fact, aliased, Val(true))
        @test aliased == ref
    end
end

@testset "RFLU keeps the stdlib path above the cutoff" begin
    n = CUTOFF + 1
    A = pivoting_matrix(Float64, n)
    b = rand(n)
    fact = rf_lu(A)

    ref = similar(b)
    ldiv!(ref, fact, b)

    u = similar(b)
    RFExt._rf_ldiv!(u, fact, b, Val(true))
    @test u == ref

    U = similar(b, n, 1)
    RFExt._rf_ldiv!(U, fact, reshape(b, n, 1), Val(true))
    @test vec(U) == ref
end

# The naive kernel is `@inbounds` and assumes a square factorization, so a
# non-square `LU` has to stay on the stdlib path (which reports the mismatch)
# rather than walking off the end of `factors`.
@testset "RFLU non-square factorization keeps the stdlib path" begin
    fact = rf_lu(rand(6, 4))
    @test_throws DimensionMismatch RFExt._rf_ldiv!(zeros(6), fact, rand(6), Val(true))
end

@testset "RFLU cached re-solve with a new b" begin
    n = 20
    A0 = pivoting_matrix(Float64, n)
    cache = init(
        LinearProblem(copy(A0), rand(n)), RFLUFactorization();
        alias = LinearAliasSpecifier(alias_A = false, alias_b = false)
    )
    for _ in 1:10
        bnew = rand(n)
        cache.b = bnew
        solve!(cache)
        @test A0 * cache.u ≈ bnew rtol = 1.0e-10
    end
end
