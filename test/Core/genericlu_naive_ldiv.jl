using LinearSolve, LinearAlgebra, Test, Random

Random.seed!(1145)

@testset "GenericLU naive back-solve correctness" begin
    for T in (Float64, Float32, ComplexF64, ComplexF32, BigFloat)
        @testset "$T" begin
            rtol = 100 * eps(float(real(one(T))))
            for n in (2, 3, 8, 20, 50, 80)
                A = rand(T, n, n) + n * I
                b = rand(T, n)
                B = rand(T, n, 4)
                xref = A \ b
                Xref = A \ B

                sol = solve(LinearProblem(copy(A), copy(b)), GenericLUFactorization())
                @test SciMLBase.successful_retcode(sol)
                @test sol.u ≈ xref rtol = rtol * n

                solB = solve(LinearProblem(copy(A), copy(B)), GenericLUFactorization())
                @test SciMLBase.successful_retcode(solB)
                @test solB.u ≈ Xref rtol = rtol * n
            end
        end
    end
end

@testset "GenericLU naive back-solve kernel vs getrs!" begin
    for n in (2, 3, 8, 16, 32, 64)
        A = rand(n, n) + n * I
        b = rand(n)
        B = rand(n, 3)
        F = lu(copy(A))

        b1 = copy(b)
        LinearSolve._naive_lu_ldiv!(F.factors, F.ipiv, b1)
        b2 = copy(b)
        ldiv!(F, b2)
        @test b1 ≈ b2 rtol = 1.0e-14

        B1 = copy(B)
        LinearSolve._naive_lu_ldiv!(F.factors, F.ipiv, B1)
        B2 = copy(B)
        ldiv!(F, B2)
        @test B1 ≈ B2 rtol = 1.0e-14
    end
end

@testset "GenericLU back-solve reuse (many RHS, one factorization)" begin
    n = 8
    A0 = rand(n, n) + n * I
    b = rand(n)
    # alias_A=false so the original matrix is preserved for residual checks
    cache = init(
        LinearProblem(copy(A0), copy(b)),
        GenericLUFactorization();
        alias = LinearAliasSpecifier(alias_A = false, alias_b = false)
    )
    solve!(cache)
    @test A0 * cache.u ≈ b rtol = 1.0e-12

    for _ in 1:20
        bnew = rand(n)
        cache.b = bnew
        solve!(cache)
        @test A0 * cache.u ≈ bnew rtol = 1.0e-12
    end
end

@testset "GenericLU naive back-solve with NoPivot" begin
    n = 6
    # Diagonally dominant so NoPivot is stable
    A = rand(n, n) + n * I
    b = rand(n)
    sol = solve(LinearProblem(copy(A), copy(b)), GenericLUFactorization(NoPivot()))
    @test sol.u ≈ A \ b rtol = 1.0e-12
end

@testset "GenericLU keeps the naive kernel at every size (no BLAS deferral)" begin
    # Path discrimination is against the `ldiv!` reference, whose bits are
    # stable across call contexts (LAPACK ccall / plain-op generic path): a
    # size-based deferral would make `sol.u` bitwise-equal to it, while the
    # naive kernel differs in the last ulps at these sizes. Comparing bitwise
    # against the kernel itself is not portable — its `muladd`/`@simd` loops
    # may contract differently standalone vs inlined (observed on Julia 1.11).
    # N = 600 exceeds both cutoffs (256 `Matrix` / 512 wrapped) in
    # `_naive_ldiv_cutoff`, which `_generic_lu_ldiv!` must never consult.
    for (mk, n) in ((identity, 600), (adjoint, 600))
        A = rand(n, n) + n * I
        b = rand(n)
        cache = init(LinearProblem(mk(copy(A)), copy(b)), GenericLUFactorization())
        sol = solve!(cache)
        F = cache.cacheval.fact
        xldiv = copy(b)
        ldiv!(F, xldiv)
        @test sol.u != xldiv
        @test sol.u ≈ Matrix(mk(A)) \ b rtol = 1.0e-10 * n

        B = rand(n, 3)
        solB = solve(LinearProblem(mk(copy(A)), copy(B)), GenericLUFactorization())
        @test solB.u ≈ Matrix(mk(A)) \ B rtol = 1.0e-10 * n
    end
end

@testset "LUFactorization back-solve is size-aware (naive below cutoff, getrs! above)" begin
    # The branch is read off `_use_naive_lu_ldiv`, not inferred from
    # `sol.u != ldiv!(...)`: the two agree to within an ulp and on a given draw
    # can agree exactly (0.13% of draws at N = 64 on an EPYC 7502 / OpenBLAS
    # 0.3.29 build), which is how the single-draw form of this check failed.
    cutoff = LinearSolve._naive_ldiv_cutoff(zeros(0, 0))
    for n in (cutoff ÷ 4, cutoff, cutoff + 1, 2 * cutoff)
        naive_expected = n <= cutoff
        A = rand(n, n) + n * I
        b = rand(n)
        cache = init(LinearProblem(copy(A), copy(b)), LUFactorization())
        u = copy(solve!(cache).u)
        F = cache.cacheval
        @test LinearSolve._use_naive_lu_ldiv(u, F, b) == naive_expected
        @test u ≈ A \ b rtol = 1.0e-10 * n
        if !naive_expected
            @test u == ldiv!(similar(b), F, copy(b))
        end
    end
    # Matrix right-hand sides keep the BLAS-3 path at every size
    n = 100
    A = rand(n, n) + n * I
    B = rand(n, 4)
    cacheB = init(LinearProblem(copy(A), copy(B)), LUFactorization())
    solB = solve!(cacheB)
    FB = cacheB.cacheval
    @test !LinearSolve._use_naive_lu_ldiv(solB.u, FB, B)
    @test solB.u == ldiv!(similar(B), FB, copy(B))
    @test solB.u ≈ A \ B rtol = 1.0e-8
end

@testset "GenericLU back-solve on Adjoint/Transpose operators" begin
    for T in (Float64, Float32, ComplexF64, ComplexF32, BigFloat),
            wrap in (adjoint, transpose)

        @testset "$wrap $T" begin
            rtol = 100 * eps(float(real(one(T))))
            sizes = T === BigFloat ? (4, 16, 33) : (4, 16, 64, 96, 128)
            for n in sizes
                P = rand(T, n, n) + n * I
                A = wrap(P)
                b = rand(T, n)
                B = rand(T, n, 3)
                xref = Matrix(A) \ b
                Xref = Matrix(A) \ B

                cache = init(
                    LinearProblem(wrap(copy(P)), copy(b)), GenericLUFactorization()
                )
                sol = solve!(cache)
                @test SciMLBase.successful_retcode(sol)
                @test sol.u ≈ xref rtol = rtol * n

                F = cache.cacheval.fact
                @test F.factors isa (wrap === adjoint ? Adjoint : Transpose)

                x1 = copy(b)
                LinearSolve._naive_lu_ldiv!(F.factors, F.ipiv, x1)
                x2 = copy(b)
                ldiv!(F, x2)
                @test x1 ≈ x2 rtol = rtol * n

                bnew = rand(T, n)
                cache.b = bnew
                sol2 = solve!(cache)
                @test sol2.u ≈ Matrix(A) \ bnew rtol = rtol * n

                solB = solve(
                    LinearProblem(wrap(copy(P)), copy(B)), GenericLUFactorization()
                )
                @test SciMLBase.successful_retcode(solB)
                @test solB.u ≈ Xref rtol = rtol * n
            end
        end
    end
end

@testset "Adjoint/Transpose strided factors dispatch to the specialized kernel" begin
    ipivT = Vector{LinearAlgebra.BlasInt}
    for W in (Adjoint, Transpose), rhsT in (Vector{Float64}, Matrix{Float64})
        @test which(
            LinearSolve._naive_lu_ldiv!,
            Tuple{W{Float64, Matrix{Float64}}, ipivT, rhsT}
        ) !== which(
            LinearSolve._naive_lu_ldiv!,
            Tuple{Matrix{Float64}, ipivT, rhsT}
        )
    end
end

@testset "smart helper is specialized only for BLAS-eligible strided vector solves" begin
    ipivT = Vector{LinearAlgebra.BlasInt}
    for T in (Float64, ComplexF32)
        luT = LU{T, Matrix{T}, ipivT}
        @test which(
            LinearSolve._smart_lu_ldiv!, Tuple{Vector{T}, luT, Vector{T}}
        ) !== which(
            LinearSolve._smart_lu_ldiv!, Tuple{Matrix{T}, luT, Matrix{T}}
        )
    end
    # No measured BLAS crossover for BigFloat: it keeps `_ldiv!` unconditionally
    luB = LU{BigFloat, Matrix{BigFloat}, Vector{Int}}
    @test which(
        LinearSolve._smart_lu_ldiv!, Tuple{Vector{BigFloat}, luB, Vector{BigFloat}}
    ) === which(
        LinearSolve._smart_lu_ldiv!, Tuple{Matrix{BigFloat}, luB, Matrix{BigFloat}}
    )
end
