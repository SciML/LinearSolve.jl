using LinearSolve, ConjugateGradients, LinearAlgebra, Test, Random

# Wrapper over ConjugateGradients.jl, which provides CG and BiCGStab.
# See SciML/LinearSolve.jl#567.
@testset "ConjugateGradients.jl (#567)" begin
    Random.seed!(567)
    n = 40
    # Symmetric positive definite, which is what CG requires.
    Aspd = let M = rand(n, n)
        M' * M + n * I
    end
    b = rand(n)
    reference = Aspd \ b

    @testset "the extension is loaded" begin
        @test Base.get_extension(LinearSolve, :LinearSolveConjugateGradientsExt) !==
            nothing
    end

    @testset "solves" begin
        for alg in (ConjugateGradientsJL_CG(), ConjugateGradientsJL_BICGSTAB())
            sol = solve(LinearProblem(Aspd, b), alg)
            @test sol.retcode == LinearSolve.ReturnCode.Success
            @test sol.u ≈ reference rtol = 1.0e-6
            @test norm(Aspd * sol.u - b) < 1.0e-6
            @test sol.iters > 0
        end
    end

    @testset "the generic constructor picks the solver" begin
        @test ConjugateGradientsJL_CG().solver === :cg
        @test ConjugateGradientsJL_BICGSTAB().solver === :bicgstab
        sol = solve(LinearProblem(Aspd, b), ConjugateGradientsJL(solver = :bicgstab))
        @test sol.retcode == LinearSolve.ReturnCode.Success
        @test sol.u ≈ reference rtol = 1.0e-6
    end

    # BiCGStab does not need symmetry, so it also handles a general system.
    @testset "BiCGStab on a nonsymmetric system" begin
        Agen = rand(n, n) + n * I
        bgen = rand(n)
        sol = solve(LinearProblem(Agen, bgen), ConjugateGradientsJL_BICGSTAB())
        @test sol.retcode == LinearSolve.ReturnCode.Success
        @test sol.u ≈ Agen \ bgen rtol = 1.0e-6
    end

    # The cacheval holds the solver's working vectors, so a re-solve has to keep
    # giving the right answer rather than reusing stale state.
    @testset "cached re-solves" begin
        cache = init(LinearProblem(copy(Aspd), copy(b)), ConjugateGradientsJL_CG())
        @test solve!(cache).u ≈ reference rtol = 1.0e-6

        b2 = rand(n)
        cache.b = b2
        @test solve!(cache).u ≈ Aspd \ b2 rtol = 1.0e-6

        A2 = let M = rand(n, n)
            M' * M + 2n * I
        end
        cache.A = A2
        @test solve!(cache).u ≈ A2 \ b2 rtol = 1.0e-6
    end

    # ConjugateGradients.jl defines its solvers for `Vector{<:Real}` only, so the
    # unsupported cases are rejected where the algorithm was chosen instead of
    # surfacing as a `MethodError` from inside the solver.
    @testset "unsupported element and array types are rejected" begin
        @test_throws ArgumentError init(
            LinearProblem(Aspd, rand(ComplexF64, n)), ConjugateGradientsJL_CG()
        )
        @test_throws ArgumentError init(
            LinearProblem(Aspd, view(rand(n + 5), 1:n)), ConjugateGradientsJL_CG()
        )
        err = try
            init(LinearProblem(Aspd, rand(ComplexF64, n)), ConjugateGradientsJL_CG())
            nothing
        catch e
            e
        end
        # The message should say what to use instead.
        @test occursin("KrylovJL", err.msg)
    end

    @testset "a right preconditioner is refused rather than ignored" begin
        cache = init(
            LinearProblem(Aspd, b), ConjugateGradientsJL_CG();
            Pr = Diagonal(fill(2.0, n))
        )
        @test_throws ArgumentError solve!(cache)
    end

    @testset "a left preconditioner is applied" begin
        sol = solve(
            LinearProblem(Aspd, b), ConjugateGradientsJL_CG();
            Pl = Diagonal(diag(Aspd))
        )
        @test sol.retcode == LinearSolve.ReturnCode.Success
        @test sol.u ≈ reference rtol = 1.0e-6
    end
end
