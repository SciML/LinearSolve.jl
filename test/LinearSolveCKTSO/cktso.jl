using LinearSolve, CKTSO
using LinearAlgebra, SparseArrays, Random, Test

# CKTSO is not redistributable, so it is only present when the runner has been pointed at
# a copy. Without one the extension never loads and there is nothing to exercise.
if !CKTSO.is_available()
    @info "CKTSO library not configured; set CKTSO_LIBRARY to run these tests"
    @testset "CKTSO unavailable" begin
        @test_throws ErrorException CKTSO.library()
    end
else
    Random.seed!(392)

    @testset "solves a sparse system" begin
        n = 60
        A = sprand(n, n, 0.12) + 3I
        b = rand(n)
        sol = solve(LinearProblem(A, b), CKTSOFactorization())
        @test SciMLBase.successful_retcode(sol)
        @test sol.u ≈ Matrix(A) \ b rtol = 1.0e-9
    end

    # The reason to reach for CKTSO: hold the symbolic analysis and refactorize the same
    # pattern with new values.
    @testset "reuses the analysis across solves" begin
        n = 50
        A = sprand(n, n, 0.15) + 3I
        b = rand(n)
        cache = init(LinearProblem(copy(A), copy(b)), CKTSOFactorization())
        @test solve!(cache).u ≈ Matrix(A) \ b rtol = 1.0e-9

        B = copy(A)
        nonzeros(B) .= nonzeros(A) .* (1 .+ rand(nnz(A)))
        cache.A = B
        @test solve!(cache).u ≈ Matrix(B) \ b rtol = 1.0e-9

        b2 = rand(n)
        cache.b = b2
        @test solve!(cache).u ≈ Matrix(B) \ b2 rtol = 1.0e-9

        # A different pattern cannot reuse the analysis, so the solve has to fall back to
        # a fresh one rather than refactorizing against the wrong symbolic structure.
        C = copy(B)
        C[1, n] = 1.0
        cache.A = C
        @test solve!(cache).u ≈ Matrix(C) \ b2 rtol = 1.0e-9
    end

    @testset "a singular matrix reports rather than throws" begin
        for M in (
                sparse([1.0 0.0; 0.0 0.0]),     # empty column
                sparse([0.0 0.0; 1.0 1.0]),     # zero row
                sparse([1.0 1.0; 1.0 1.0]),     # numerically singular
            )
            sol = solve(LinearProblem(M, [1.0, 1.0]), CKTSOFactorization())
            @test !SciMLBase.successful_retcode(sol)
        end
    end

    @testset "rejects a nonsquare matrix" begin
        @test_throws ErrorException solve(
            LinearProblem(sprand(6, 4, 0.5), rand(6)), CKTSOFactorization()
        )
    end
end
