using FastAlmostBandedMatrices, BandedMatrices, LinearAlgebra, LinearSolve, Test

# Square Case
n = 8
A = BandedMatrix(Matrix(I, n, n), (2, 2))
b = ones(n)
A1 = A / 1;
b1 = rand(n);
x1 = zero(b);
A2 = A / 2;
b2 = rand(n);
x2 = zero(b);

sol1 = solve(LinearProblem(A1, b1; u0 = x1))
@test sol1.u ≈ A1 \ b1
sol2 = solve(LinearProblem(A2, b2; u0 = x2))
@test sol2.u ≈ A2 \ b2

A = AlmostBandedMatrix(BandedMatrix(fill(2.0, n, n), (1, 1)), fill(3.0, 2, n))
A[band(0)] .+= 1:n

sol1ab = solve(LinearProblem(A, b; u0 = x1))
@test sol1ab.u ≈ Matrix(A) \ b

# Square Symmetric
A1s = Symmetric(A1)
A2s = Symmetric(A2)

sol1s = solve(LinearProblem(A1s, b1; u0 = x1))
@test sol1s.u ≈ A1s \ b1
sol2s = solve(LinearProblem(A2s, b2; u0 = x2))
@test sol2s.u ≈ A2s \ b2

# Underdetermined. BandedMatrices can factor a wide matrix but `\` on the result
# throws "Not implemented", so LinearSolve solves through the QR of `Aᵀ`, which is
# banded too. The answer is the minimum-norm one, matching dense LAPACK. See #419.
@testset "Underdetermined BandedMatrix (#419)" begin
    for (m, n, l, u) in ((8, 10, 2, 2), (6, 10, 2, 1), (3, 12, 0, 2), (9, 10, 3, 2))
        Aud = BandedMatrix(rand(m, n), (l, u))
        bud = rand(m)
        reference = Matrix(Aud) \ bud

        for alg in (nothing, QRFactorization())
            sol = alg === nothing ? solve(LinearProblem(Aud, bud)) :
                solve(LinearProblem(Aud, bud), alg)
            @test sol.retcode == LinearSolve.ReturnCode.Success
            @test length(sol.u) == n
            @test Aud * sol.u ≈ bud
            # Of the infinitely many solutions, the minimum-norm one.
            @test sol.u ≈ reference
            @test norm(sol.u) <= norm(reference) + 1.0e-8
        end
    end

    # The cached path factors once and reuses it, so check a re-solve too.
    Aud = BandedMatrix(rand(6, 10), (2, 1))
    bud1 = rand(6)
    cache = init(LinearProblem(Aud, bud1), QRFactorization())
    @test solve!(cache).u ≈ Matrix(Aud) \ bud1
    bud2 = rand(6)
    cache.b = bud2
    @test solve!(cache).u ≈ Matrix(Aud) \ bud2
end

# `AlmostBandedMatrix` hits the same wall, and is solved the same way, except its
# transpose cannot stay structured: banded plus dense fill rows transposes to
# dense fill columns, so the QR of the transpose densifies. The answer is still
# the minimum-norm one, so it agrees with the `BandedMatrix` path above.
@testset "Underdetermined AlmostBandedMatrix (#419)" begin
    for (m, k, l, u, nfill) in ((n - 2, n, 1, 1, 2), (5, 10, 2, 1, 3), (4, 9, 1, 2, 2))
        Aud = AlmostBandedMatrix(BandedMatrix(fill(2.0, m, k), (l, u)), rand(nfill, k))
        Aud[band(0)] .+= 1:m
        bud = rand(m)
        reference = Matrix(Aud) \ bud

        for alg in (nothing, QRFactorization())
            sol = alg === nothing ? solve(LinearProblem(Aud, bud)) :
                solve(LinearProblem(Aud, bud), alg)
            @test sol.retcode == LinearSolve.ReturnCode.Success
            @test length(sol.u) == k
            @test Matrix(Aud) * sol.u ≈ bud
            @test sol.u ≈ reference
            @test norm(sol.u) <= norm(reference) + 1.0e-8
        end
    end
end

# The R of a banded QR has upper bandwidth `l + u`, so factoring in place into
# `A`'s own `(l, u)` storage drops the fill. That used to return a wrong answer
# for square systems and a non-least-squares one for overdetermined ones, with no
# error or warning. See #1202.
@testset "Banded QR keeps the l+u fill (#1202)" begin
    for (m, k, l, u) in (
            (10, 6, 2, 1), (20, 8, 3, 2), (30, 11, 4, 3),
            (10, 6, 2, 0), (10, 6, 0, 2),
        )
        Aod = BandedMatrix(brand(m, k, l, u))
        bod = rand(m)
        reference = Matrix(Aod) \ bod
        sol = solve(LinearProblem(Aod, bod), QRFactorization())
        @test sol.retcode == LinearSolve.ReturnCode.Success
        @test sol.u ≈ reference
        # The least-squares optimum, not merely some solution.
        @test norm(Aod * sol.u - bod) <= norm(Aod * reference - bod) + 1.0e-10
    end

    # Square banded reaches QR only when asked for explicitly, and was wrong
    # outright there rather than just suboptimal.
    for (k, l, u) in ((8, 2, 1), (15, 3, 2))
        Asq = BandedMatrix(brand(k, k, l, u))
        Asq[band(0)] .+= k
        bsq = rand(k)
        sol = solve(LinearProblem(Asq, bsq), QRFactorization())
        @test sol.retcode == LinearSolve.ReturnCode.Success
        @test sol.u ≈ Matrix(Asq) \ bsq
        @test norm(Asq * sol.u - bsq) < 1.0e-10
    end

    # `inplace = true` is the default and must agree with the explicit opt-out.
    Aip = BandedMatrix(brand(12, 7, 2, 2))
    bip = rand(12)
    @test solve(LinearProblem(Aip, bip), QRFactorization()).u ≈
        solve(LinearProblem(Aip, bip), QRFactorization(NoPivot(), false)).u
end

# Overdetermined
A = BandedMatrix(ones(10, 8), (2, 0))
b = rand(10)

@test_nowarn solve(LinearProblem(A, b))

A = AlmostBandedMatrix(BandedMatrix(fill(2.0, n + 2, n), (1, 1)), fill(3.0, 2, n))
A[band(0)] .+= 1:n

@test_nowarn solve(LinearProblem(A, b))

# Singular BandedMatrix - should gracefully fall back to pivoted QR instead of throwing
A_singular = BandedMatrix(zeros(n, n), (2, 2))
A_singular[band(0)] .= [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]
b_singular = ones(n)
sol_singular = solve(LinearProblem(A_singular, b_singular))
@test sol_singular.retcode == LinearSolve.ReturnCode.Success

# Workaround for no lu from BandedMatrices
A = BandedMatrix{BigFloat}(ones(3, 3), (0, 0))
b = BigFloat[1, 2, 3]
prob = LinearProblem(A, b)
@test_nowarn solve(prob)
