# Internal tests for the vendored SupernodalQR solver (src/SupernodalQR):
# the pure-Julia multifrontal sparse Householder QR.  The LinearSolve-level
# algorithm surface is covered at the end; the rest exercises the solver's own
# invariants.

using LinearSolve, SparseArrays, LinearAlgebra, Random, Test, SciMLBase
const SNQR = LinearSolve.SupernodalQR

Random.seed!(42)

function poisson2d(k)
    n = k * k
    Is = Int[]; Js = Int[]; V = Float64[]
    idx(i, j) = (j - 1) * k + i
    for j in 1:k, i in 1:k
        c = idx(i, j)
        push!(Is, c); push!(Js, c); push!(V, 4.0)
        i > 1 && (push!(Is, c); push!(Js, idx(i - 1, j)); push!(V, -1.0))
        i < k && (push!(Is, c); push!(Js, idx(i + 1, j)); push!(V, -1.0))
        j > 1 && (push!(Is, c); push!(Js, idx(i, j - 1)); push!(V, -1.0))
        j < k && (push!(Is, c); push!(Js, idx(i, j + 1)); push!(V, -1.0))
    end
    return sparse(Is, Js, V, n, n)
end

# full-rank tall matrix with a random sparse part
tallmat(T, m, n, d) = sprand(T, m, n, d) + [sparse(one(T) * I, n, n); spzeros(T, m - n, n)]

@testset "R factor: R'R = A[:,q]'A[:,q]" begin
    for A in (tallmat(Float64, 200, 120, 0.03), poisson2d(15), tallmat(ComplexF64, 90, 60, 0.05))
        for ordering in (:amd, :natural)
            # singletons stay rows of A and are not part of F.R
            F = SNQR.snqr(A; ordering = ordering, singletons = false)
            Aq = Matrix(A[:, F.q])
            R = Matrix(F.R)
            @test istriu(R)
            @test norm(R' * R - Aq' * Aq) <= 1.0e-13 * norm(Aq)^2
            @test rank(F) == size(A, 2)
        end
    end
end

@testset "least squares (tall/square)" begin
    for A in (tallmat(Float64, 200, 120, 0.03), poisson2d(20), [poisson2d(10); 2poisson2d(10)])
        b = randn(size(A, 1))
        x = SNQR.snqr(A) \ b
        @test x ≈ Matrix(A) \ b rtol = 1.0e-10
    end
    A = tallmat(ComplexF64, 150, 90, 0.05)
    b = randn(ComplexF64, 150)
    @test SNQR.snqr(A) \ b ≈ Matrix(A) \ b rtol = 1.0e-10
end

@testset "minimum norm (wide = :minnorm)" begin
    for T in (Float64, ComplexF64)
        A = copy(tallmat(T, 200, 80, 0.05)')
        b = randn(T, 80)
        F = SNQR.snqr(A; wide = :minnorm)
        @test F.sym.transposed
        x = F \ b
        @test A * x ≈ b
        @test x ≈ pinv(Matrix(A)) * b rtol = 1.0e-10
    end
    # rank-deficient wide (repeated rows) with an inconsistent right-hand side:
    # the dead-row correction must still give the minimum-norm least-squares
    # solution, for one and for several right-hand sides
    W = sprand(40, 90, 0.08)
    W = [W; W[1:5, :]; 2 .* W[6:7, :]]
    F = SNQR.snqr(W; wide = :minnorm)
    @test rank(F) == rank(Matrix(W))
    b = randn(47)
    @test F \ b ≈ pinv(Matrix(W)) * b rtol = 1.0e-8
    B = randn(47, 3)
    @test F \ B ≈ pinv(Matrix(W)) * B rtol = 1.0e-8
end

@testset "basic solutions (wide) and column singletons" begin
    # wide, full row rank: an exact basic solution, at most rank(A) nonzeros
    for T in (Float64, ComplexF64)
        A = copy(tallmat(T, 200, 80, 0.05)')
        b = randn(T, 80)
        F = SNQR.snqr(A)
        @test !F.sym.transposed
        x = F \ b
        @test A * x ≈ b
        @test count(!iszero, x) <= rank(F) == 80
    end
    # wide, rank deficient, inconsistent: least squares (normal equations)
    W = sprand(40, 90, 0.08)
    W = [W; W[1:5, :]; 2 .* W[6:7, :]]
    F = SNQR.snqr(W)
    @test rank(F) == rank(Matrix(W))
    b = randn(47)
    x = F \ b
    @test norm(W' * (W * x - b)) <= 1.0e-10 * norm(W' * b)
    @test count(!iszero, x) <= rank(F)
    # LP-style [N I]: the slack columns are singletons and cover every row
    N = sprand(60, 150, 0.05)
    A = [N sparse(2.0I, 60, 60)]
    F = SNQR.snqr(A)
    @test length(F.sym.scol) == 60 && F.sym.nmf == 0 && length(F.sym.ecol) == 150
    b = randn(60)
    @test A * (F \ b) ≈ b
    # tall with a triangular singleton block on top of a least-squares block
    U = triu(sprand(30, 30, 0.2)) + 3I
    A = [U sprand(30, 20, 0.1); spzeros(50, 30) tallmat(Float64, 50, 20, 0.1)]
    F = SNQR.snqr(A)
    @test length(F.sym.scol) >= 30
    b = randn(80)
    @test F \ b ≈ Matrix(A) \ b rtol = 1.0e-10
    B = randn(80, 3)
    @test F \ B ≈ Matrix(A) \ B rtol = 1.0e-10
    # a refactorization that zeroes a singleton pivot redoes the analysis
    A2 = copy(A)
    nonzeros(A2)[F.sym.spos[1]] = 0
    @test SNQR.snqr!(F, A2) === F
    @test !(F.sym.scol[1] in F.sym.scol[2:end])
    x = F \ b
    @test norm(A2' * (A2 * x - b)) <= 1.0e-10 * norm(A2' * b)
end

@testset "rank deficiency (Heath dead columns)" begin
    # duplicated columns: rank 60 of 70, solution satisfies the normal equations
    A = sprand(120, 60, 0.08)
    A = [A A[:, 1:10]]
    F = SNQR.snqr(A)
    @test rank(F) == rank(Matrix(A))
    b = randn(120)
    x = F \ b
    @test norm(A' * (A * x - b)) <= 1.0e-10 * norm(A' * b)

    # square, structurally singular (zero row + zero column)
    S = poisson2d(10)
    S[:, 5] .= 0
    S[7, :] .= 0
    dropzeros!(S)
    FS = SNQR.snqr(S)
    @test rank(FS) == 99
    b = randn(100)
    x = FS \ b
    @test norm(S' * (S * x - b)) <= 1.0e-10 * norm(b)
    @test x[5] == 0

    # the all-zero matrix has rank 0 and a zero solution
    Z = spzeros(5, 3)
    @test rank(SNQR.snqr(Z)) == 0
    @test SNQR.snqr(Z) \ ones(5) == zeros(3)
end

@testset "Householder kernel on subnormal residue" begin
    # cancellation can leave a column whose norm is subnormal; scaling by
    # 1/(ξ₁ + ν) would overflow and turn the zeros below into NaN
    # (g7jac100 from the SuiteSparse collection hit this)
    Fs = reshape([3.5e-323, 0.0, 5.0e-324, 0.0], 4, 1)
    τ = SNQR._house!(Fs, 1, 1, 4, norm(Fs))
    @test iszero(τ)
    @test all(isfinite, Fs)
end

@testset "dense rows, empty rows, generic eltypes" begin
    A = tallmat(Float64, 300, 100, 0.02)
    A[5, :] .= 1.0                     # dense row (ignored by the ordering only)
    A[17, :] .= 0                      # empty row
    dropzeros!(A)
    b = randn(300)
    @test SNQR.snqr(A) \ b ≈ Matrix(A) \ b rtol = 1.0e-10

    Ab = big.(tallmat(Float64, 40, 25, 0.1))
    bb = big.(randn(40))
    xb = SNQR.snqr(Ab) \ bb
    @test norm(Ab' * (Ab * xb - bb)) < 1.0e-60
end

@testset "multiple right-hand sides and refactorization" begin
    A = poisson2d(20)
    F = SNQR.snqr(A)
    B = randn(size(A, 1), 3)
    @test A * (F \ B) ≈ B
    A2 = copy(A)
    nonzeros(A2) .*= 1 .+ 0.1 .* rand(nnz(A2))
    @test SNQR.snqr!(F, A2) === F
    b = randn(size(A, 1))
    @test A2 * (F \ b) ≈ b
    @test_throws ArgumentError SNQR.snqr!(F, A2 + sparse(1:400, 400:-1:1, 1.0))
end

@testset "SupernodalQRFactorization (LinearSolve interface)" begin
    @test !LinearSolve.needs_square_A(SupernodalQRFactorization())
    A = poisson2d(12)
    b = randn(size(A, 1))
    sol = solve(LinearProblem(A, b), SupernodalQRFactorization())
    @test SciMLBase.successful_retcode(sol)
    @test A * sol.u ≈ b

    # tall least squares; wide basic (default) and minimum norm
    At = tallmat(Float64, 150, 90, 0.05)
    bt = randn(150)
    @test solve(LinearProblem(At, bt), SupernodalQRFactorization()).u ≈ Matrix(At) \ bt
    Aw = copy(At')
    bw = randn(90)
    @test Aw * solve(LinearProblem(Aw, bw), SupernodalQRFactorization()).u ≈ bw
    @test solve(LinearProblem(Aw, bw), SupernodalQRFactorization(wide = :minnorm)).u ≈
        pinv(Matrix(Aw)) * bw

    # cache reuse: same pattern (numeric refactorization) and a changed pattern
    for alg in (SupernodalQRFactorization(), SupernodalQRFactorization(reuse_symbolic = false))
        cache = SciMLBase.init(LinearProblem(A, b), alg)
        @test A * solve!(cache).u ≈ b
        cache.A = A / 2
        @test (A / 2) * solve!(cache).u ≈ b
        X = sprand(size(A)..., 0.05) + 10I
        cache.A = X
        @test X * solve!(cache).u ≈ b
    end
end
