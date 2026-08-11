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

A = AlmostBandedMatrix(BandedMatrix(fill(2.0, n - 2, n), (1, 1)), fill(3.0, 2, n))
A[band(0)] .+= 1:(n - 2)

@test_throws ErrorException solve(LinearProblem(A, b)).u

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
