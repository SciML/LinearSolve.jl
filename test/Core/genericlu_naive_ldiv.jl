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
