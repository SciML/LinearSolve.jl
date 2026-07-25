# Internal tests for the vendored SupernodalLU solver (src/SupernodalLU):
# the pure-Julia supernodal left-right-looking LU of Schenk & Gärtner.
# The LinearSolve-level algorithm surface is covered in basictests.jl and
# resolve.jl; these exercise the solver's own invariants.

using LinearSolve, SparseArrays, LinearAlgebra, Random, Test
using RecursiveFactorization        # activates the RF/TriangularSolve kernels
const SNLU = LinearSolve.SupernodalLU

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

function poisson3d(k)
    n = k^3
    Is = Int[]; Js = Int[]; V = Float64[]
    idx(i, j, l) = ((l - 1) * k + (j - 1)) * k + i
    for l in 1:k, j in 1:k, i in 1:k
        c = idx(i, j, l)
        push!(Is, c); push!(Js, c); push!(V, 6.0)
        for (di, dj, dl) in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
            ii, jj, ll = i + di, j + dj, l + dl
            if 1 <= ii <= k && 1 <= jj <= k && 1 <= ll <= k
                push!(Is, c); push!(Js, idx(ii, jj, ll)); push!(V, -1.0)
            end
        end
    end
    return sparse(Is, Js, V, n, n)
end

@testset "factor identity A[p,q] = L*U" begin
    for A in (
            sprand(60, 60, 0.15) + 10I,
            poisson2d(15),
            spdiagm(-1 => fill(-1.0, 39), 0 => fill(2.5, 40), 1 => fill(-1.3, 39)),
        )
        F = SNLU.snlu(A; matching = false)
        @test norm(Matrix(A[F.p, F.q]) - Matrix(F.L) * Matrix(F.U)) <=
            1.0e-12 * norm(Matrix(A))
        @test SNLU.nperturbed(F) == 0
        b = randn(size(A, 1))
        x = similar(b)
        SNLU.solve!(x, F, b)
        @test norm(A * x - b) <= 1.0e-11 * norm(b)
    end
end

@testset "allocation-free solve; bounded refactorization allocation" begin
    A = poisson2d(40)
    n = size(A, 1)
    b = randn(n)
    x = similar(b)
    F = SNLU.snlu(A)
    SNLU.snlu!(F, A)
    SNLU.solve!(x, F, b)
    # Solves allocate nothing.  Refactorization allocates only the small
    # `LinearSolution` object each dense block cache's `solve!` returns —
    # a fixed ~64 B per cached supernode, independent of problem size.
    @test (@allocated SNLU.solve!(x, F, b)) == 0
    ncache = count(!isnothing, F.bcaches)
    @test (@allocated SNLU.snlu!(F, A)) <= 128 * max(ncache, 1)
    # with the dense caches disabled the built-in kernel is fully allocation-free
    Fb = SNLU.snlu(A; dense_threshold = typemax(Int))
    SNLU.snlu!(Fb, A)
    @test (@allocated SNLU.snlu!(Fb, A)) == 0
end

@testset "matching engages on zero/weak diagonals" begin
    n = 40
    P = sparse(collect(n:-1:1), collect(1:n), 2.0 .+ rand(n), n, n)
    F = SNLU.snlu(P)
    @test F.matched
    b = randn(n)
    x = similar(b)
    SNLU.solve!(x, F, P * ones(n))
    @test x ≈ ones(n) rtol = 1.0e-9
    # saddle point
    m, k = 120, 40
    H = sprand(m, m, 0.05) + 10I
    H = (H + H') / 2
    B = sprand(m, k, 0.2) + 0.5 * sparse(1:k, 1:k, 1.0, m, k)
    K = [H B; B' spzeros(k, k)]
    bk = randn(m + k)
    FK = SNLU.snlu(K)
    xk = similar(bk)
    SNLU.solve!(xk, FK, bk)
    @test norm(K * xk - bk) <= 1.0e-10 * norm(bk)
end

@testset "mass-perturbation retry engages matching" begin
    n = 200
    D = 10.0 .^ range(-9, 9; length = n)
    A = spdiagm(0 => D, 1 => fill(1.0e-9, n - 1), -1 => fill(1.0e-9, n - 1))
    F = SNLU.snlu(A; check = false)
    @test F.matched
    @test SNLU.nperturbed(F) == 0
    b = randn(n)
    x = similar(b)
    SNLU.solve!(x, F, b)
    @test norm(A * x - b) <= 1.0e-9 * norm(b)
end

@testset "banded fast path keeps natural ordering" begin
    n = 4000
    bw = 6
    Is = Int[]; Js = Int[]; Vs = Float64[]
    for j in 1:n
        push!(Is, j); push!(Js, j); push!(Vs, 4.0 * bw)
        for d in 1:bw
            j + d <= n || continue
            v = randn()
            push!(Is, j + d); push!(Js, j); push!(Vs, v)
            push!(Is, j); push!(Js, j + d); push!(Vs, v)
        end
    end
    A = sparse(Is, Js, Vs, n, n)
    sym = SNLU.snlu_symbolic(A)
    @test sym.qf == 1:n
end

@testset "threaded matches serial (same pivot sequence)" begin
    A = sprand(6000, 6000, 8 / 6000) + 10I
    Fs = SNLU.snlu(A)
    Ft = SNLU.snlu(A; threaded = true)
    @test Fs.p == Ft.p
    @test Fs.nperturbed == Ft.nperturbed
    ok = true
    for s in eachindex(Fs.W)
        ok &= norm(Fs.W[s] - Ft.W[s]) <= 1.0e-10 * max(norm(Fs.W[s]), 1.0)
    end
    @test ok
end

@testset "generic eltypes" begin
    T = spdiagm(0 => fill(big"4.0", 40), -1 => fill(big"-1.0", 39), 1 => fill(big"-1.0", 39))
    b = BigFloat.(1:40)
    F = SNLU.snlu(T)
    x = similar(b)
    SNLU.solve!(x, F, b)
    @test norm(T * x - b) <= big"1e-60" * norm(b)

    C = sprand(ComplexF64, 60, 60, 0.08) + (3.0 + 1.0im) * I
    bc = randn(ComplexF64, 60)
    Fc = SNLU.snlu(C)
    xc = similar(bc)
    SNLU.solve!(xc, Fc, bc)
    @test norm(C * xc - bc) <= 1.0e-11 * norm(bc)
end

@testset "hand-rolled panel kernels agree with the BLAS path" begin
    # Below PANEL_BLAS_CUTOFF the sweeps use the in-package kernels; the two
    # paths must agree to round-off.  Compare against a factor whose panels
    # are all forced onto the BLAS path via a cutoff of 0.
    for A in (poisson2d(25), sprand(800, 800, 0.01) + 12I)
        n = size(A, 1)
        F = SNLU.snlu(A)
        b = randn(n)
        x = similar(b)
        SNLU.solve!(x, F, b; refine = 0)
        @test norm(A * x - b) <= 1.0e-11 * norm(b)
        @test norm(x - (A \ b)) <= 1.0e-9 * norm(x)
        # multi-RHS goes through the gemm path and must match column-wise
        B = randn(n, 3)
        X = similar(B)
        SNLU.solve!(X, F, B; refine = 0)
        for r in 1:3
            xr = similar(b)
            SNLU.solve!(xr, F, view(B, :, r); refine = 0)
            @test X[:, r] ≈ xr rtol = 1.0e-12
        end
    end
    # a factorization with panels wider than the cutoff exercises both sides
    A3 = poisson3d(16)
    F3 = SNLU.snlu(A3; ordering = :nd)
    widest = maximum(
        F3.sym.sstart[s + 1] - F3.sym.sstart[s] for s in 1:(length(F3.sym.sstart) - 1)
    )
    @test widest >= SNLU.PANEL_BLAS_CUTOFF     # both branches are reached
    b3 = randn(size(A3, 1))
    x3 = similar(b3)
    SNLU.solve!(x3, F3, b3; refine = 0)
    @test norm(A3 * x3 - b3) <= 1.0e-11 * norm(b3)
end

@testset "multi-RHS" begin
    A = sprand(200, 200, 0.03) + 8I
    F = SNLU.snlu(A)
    B = randn(200, 5)
    X = similar(B)
    SNLU.solve!(X, F, B)
    @test norm(A * X - B) <= 1.0e-11 * norm(B)
end
