using LinearSolve, LinearAlgebra, Test, Random
using LinearAlgebra: BlasInt

Random.seed!(1234)

_ipiv(n) = Vector{BlasInt}(undef, n)

# Backward-stability residual ||P*A - L*U|| / (||L|| ||U|| eps max(m,n)).
# Normalizing by ||L||*||U|| (not ||A||) keeps the bound valid on high-growth
# matrices, where element growth is real and shared by LAPACK.
function scaled_residual(A, F)
    m, n = size(A)
    minmn = min(m, n)
    L = [
        i > j ? F.factors[i, j] : (i == j ? one(eltype(A)) : zero(eltype(A)))
            for i in 1:m, j in 1:minmn
    ]
    U = [i <= j ? F.factors[i, j] : zero(eltype(A)) for i in 1:minmn, j in 1:n]
    PA = copy(Matrix(A))
    for k in 1:minmn
        p = F.ipiv[k]
        if p != k
            PA[k, :], PA[p, :] = PA[p, :], PA[k, :]
        end
    end
    den = opnorm(L, 1) * opnorm(U, 1) * eps(real(eltype(A))) * max(m, n)
    return opnorm(PA - L * U, 1) / den
end

@testset "blocked generic_lufact! kernel" begin
    @testset "dispatch routes float strided matrices to the blocked method" begin
        m_blocked = which(
            LinearSolve.generic_lufact!,
            Tuple{Matrix{Float64}, RowMaximum, Vector{BlasInt}}
        )
        @test endswith(String(m_blocked.file), "blocked_lufact.jl")
        m_generic = which(
            LinearSolve.generic_lufact!,
            Tuple{Matrix{BigFloat}, RowMaximum, Vector{BlasInt}}
        )
        @test !endswith(String(m_generic.file), "blocked_lufact.jl")
    end

    @testset "residual, square, default params ($T)" for T in (Float64, Float32)
        for n in (
                1, 2, 3, 5, 7, 8, 9, 13, 16, 17, 31, 32, 33, 40, 41, 63, 64, 65,
                100, 127, 128, 129, 200, 250, 256, 257, 300, 500, 512,
            )
            A = randn(T, n, n)
            F = LinearSolve.generic_lufact!(copy(A), RowMaximum(), _ipiv(n); check = false)
            @test F.info == 0
            @test scaled_residual(A, F) < 20
        end
    end

    @testset "forced blocked driver, remainder paths" begin
        # panel/rowblock combinations that exercise the 4-wide remainders in
        # the trsm and both Schur kernels, and the packed/unpacked switch
        for n in (11, 41, 67, 70, 97, 130, 190, 257), nb in (4, 5, 8, 13, 16, 32),
                rb in (7, 64, 384)

            n <= nb && continue
            A = randn(n, n)
            W = copy(A)
            ipiv = _ipiv(n)
            info = LinearSolve._blocked_lufact!(W, ipiv, n, n, n, nb, rb)
            F = LU{Float64, Matrix{Float64}, Vector{BlasInt}}(W, ipiv, BlasInt(info))
            @test scaled_residual(A, F) < 20
        end
    end

    @testset "rectangular m x n" begin
        for (m, n) in (
                (100, 3), (3, 100), (128, 40), (40, 128), (257, 130),
                (130, 257), (65, 64), (64, 65), (513, 100), (100, 513),
            )
            A = randn(m, n)
            F = LinearSolve.generic_lufact!(
                copy(A), RowMaximum(), _ipiv(min(m, n)); check = false
            )
            @test scaled_residual(A, F) < 20
        end
    end

    @testset "strided views (contiguous and non-unit row stride)" begin
        for n in (17, 80)
            P = randn(2n + 3, n + 2)
            V1 = @view P[2:(n + 1), 2:(n + 1)]
            A1 = copy(Matrix(V1))
            F1 = LinearSolve.generic_lufact!(V1, RowMaximum(), _ipiv(n); check = false)
            @test scaled_residual(A1, F1) < 20
            V2 = @view P[1:2:(2n - 1), 1:n]
            A2 = copy(Matrix(V2))
            F2 = LinearSolve.generic_lufact!(V2, RowMaximum(), _ipiv(n); check = false)
            @test scaled_residual(A2, F2) < 20
        end
    end

    @testset "pivot sequence matches LAPACK when margins are decisive" begin
        for n in (17, 64, 129, 300), trial in 1:3
            # rows of widely separated magnitude, then shuffled: every pivot
            # decision has orders-of-magnitude margin, so blocked-vs-unblocked
            # rounding differences cannot flip it and ipiv must match getrf
            mags = shuffle!([2.0^k for k in 1:n])
            A = Diagonal(mags) * (I + 0.01 .* randn(n, n))
            A = A[shuffle(1:n), :]
            Fl = lu(copy(A); check = false)
            Fs = LinearSolve.generic_lufact!(copy(A), RowMaximum(), _ipiv(n); check = false)
            @test Fs.ipiv == Fl.ipiv
            @test scaled_residual(A, Fs) < 20
        end
    end

    @testset "permutation matrix input reproduces LAPACK exactly" begin
        for n in (16, 65, 200)
            p = shuffle(1:n)
            A = Matrix{Float64}(I, n, n)[p, :]
            Fl = lu(copy(A); check = false)
            Fs = LinearSolve.generic_lufact!(copy(A), RowMaximum(), _ipiv(n); check = false)
            @test Fs.ipiv == Fl.ipiv
            @test Fs.factors == Fl.factors
            @test Fs.info == 0
        end
    end

    @testset "growth matrix (Wilkinson 2^(n-1) growth)" begin
        for n in (24, 53)
            A = Matrix{Float64}(I, n, n) .- tril(ones(n, n), -1)
            A[:, n] .= 1.0
            Fs = LinearSolve.generic_lufact!(copy(A), RowMaximum(), _ipiv(n); check = false)
            @test all(Fs.ipiv .== 1:n)
            @test Fs.factors[n, n] == 2.0^(n - 1)
            @test scaled_residual(A, Fs) < 20
        end
    end

    @testset "singularity detection and check semantics" begin
        for n in (10, 50, 130), zc in (1, 4, n)
            A = randn(n, n)
            A[:, zc] .= 0.0
            Fl = lu(copy(A); check = false)
            Fs = LinearSolve.generic_lufact!(copy(A), RowMaximum(), _ipiv(n); check = false)
            @test Fs.info > 0
            @test !LinearAlgebra.issuccess(Fs)
            @test Fs.info == Fl.info
        end
        Z = zeros(50, 50)
        Fz = LinearSolve.generic_lufact!(copy(Z), RowMaximum(), _ipiv(50); check = false)
        @test Fz.info == 1
        @test_throws SingularException LinearSolve.generic_lufact!(
            copy(Z), RowMaximum(), _ipiv(50)
        )
        Fa = LinearSolve.generic_lufact!(
            copy(Z), RowMaximum(), _ipiv(50); allowsingular = true
        )
        @test Fa.info == 1
        # chkfinite runs before factorizing when check=true, like the scalar path
        An = randn(30, 30)
        An[2, 2] = NaN
        @test_throws ArgumentError LinearSolve.generic_lufact!(
            copy(An), RowMaximum(), _ipiv(30)
        )
        Fn = LinearSolve.generic_lufact!(copy(An), RowMaximum(), _ipiv(30); check = false)
        @test any(isnan, Fn.factors)
    end

    @testset "ipiv too short throws" begin
        @test_throws ArgumentError LinearSolve.generic_lufact!(
            randn(5, 5), RowMaximum(), _ipiv(3); check = false
        )
    end
end

@testset "GenericLUFactorization through the blocked kernel" begin
    @testset "solve correctness across the size cutover ($T)" for T in (Float64, Float32)
        rtol = 100 * eps(real(one(T)))
        for n in (2, 4, 8, 9, 16, 51, 100, 257)
            A = rand(T, n, n) + n * I
            b = rand(T, n)
            B = rand(T, n, 4)
            sol = solve(LinearProblem(copy(A), copy(b)), GenericLUFactorization())
            @test SciMLBase.successful_retcode(sol)
            @test sol.u ≈ A \ b rtol = rtol * n
            solB = solve(LinearProblem(copy(A), copy(B)), GenericLUFactorization())
            @test SciMLBase.successful_retcode(solB)
            @test solB.u ≈ A \ B rtol = rtol * n
        end
    end

    @testset "repeated refactorization via cache reuse" begin
        n = 100
        b = rand(n)
        cache = init(LinearProblem(rand(n, n) + n * I, b), GenericLUFactorization())
        Awork = cache.A
        for _ in 1:3
            A = rand(n, n) + n * I
            # the cache factorizes its matrix in place, so hand it a copy
            copyto!(Awork, A)
            cache.A = Awork
            sol = solve!(cache)
            @test sol.retcode == ReturnCode.Success
            @test sol.u ≈ A \ b rtol = 1.0e-10
        end
    end

    @testset "singular matrix returns Failure retcode" begin
        for n in (6, 60)
            A = zeros(n, n)
            sol = solve(LinearProblem(A, rand(n)), GenericLUFactorization())
            @test sol.retcode == ReturnCode.Failure
        end
    end
end
