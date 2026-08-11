using LinearSolve, SparseArrays, LinearAlgebra, Test, Random

# `UMFPACKFactorization` exposes UMFPACK's iterative refinement through
# `max_iterative_refinement_steps`. SuiteSparse defaults it to 2; Julia's
# SparseArrays turns it off (JuliaLang/julia#122), and `nothing` keeps whatever
# SparseArrays defaults to. See SciML/LinearSolve.jl#383.
@static if Base.USE_GPL_LIBS
    using SparseArrays.UMFPACK: JL_UMFPACK_IRSTEP

    # The control entry actually handed to UMFPACK for this cache.
    cache_irstep(cache) = cache.cacheval.control[JL_UMFPACK_IRSTEP]

    Random.seed!(383)
    # Diagonally dominant apart from one tiny pivot, so the solve is accurate
    # enough to be meaningful but ill conditioned enough for refinement to bite.
    n = 60
    A = sprand(n, n, 0.25) + n * I
    A[1, 1] = 1.0e-9
    xtrue = rand(n)
    b = A * xtrue
    relerr(u) = norm(u - xtrue) / norm(xtrue)

    @testset "UMFPACKFactorization iterative refinement (#383)" begin
        @testset "construction" begin
            @test UMFPACKFactorization().max_iterative_refinement_steps === nothing
            @test UMFPACKFactorization(
                max_iterative_refinement_steps = 2
            ).max_iterative_refinement_steps == 2
            # The other keywords keep working alongside it.
            alg = UMFPACKFactorization(
                reuse_symbolic = false, max_iterative_refinement_steps = 0
            )
            @test alg.reuse_symbolic == false
            @test alg.max_iterative_refinement_steps == 0
            @test_throws ArgumentError UMFPACKFactorization(
                max_iterative_refinement_steps = -1
            )
        end

        @testset "the setting reaches UMFPACK" begin
            for steps in (0, 1, 2, 4)
                cache = init(
                    LinearProblem(copy(A), copy(b)),
                    UMFPACKFactorization(max_iterative_refinement_steps = steps)
                )
                sol = solve!(cache)
                @test SciMLBase.successful_retcode(sol)
                @test cache_irstep(cache) == steps
            end
        end

        @testset "default leaves SparseArrays' own default in place" begin
            plain = init(LinearProblem(copy(A), copy(b)), UMFPACKFactorization())
            solve!(plain)
            # SparseArrays disables refinement, so the default must match an
            # explicit 0 rather than pinning a value of its own.
            explicit = init(
                LinearProblem(copy(A), copy(b)),
                UMFPACKFactorization(max_iterative_refinement_steps = 0)
            )
            solve!(explicit)
            @test cache_irstep(plain) == cache_irstep(explicit)
        end

        @testset "refinement improves accuracy" begin
            off = solve(
                LinearProblem(copy(A), copy(b)),
                UMFPACKFactorization(max_iterative_refinement_steps = 0)
            )
            on = solve(
                LinearProblem(copy(A), copy(b)),
                UMFPACKFactorization(max_iterative_refinement_steps = 2)
            )
            @test SciMLBase.successful_retcode(off)
            @test SciMLBase.successful_retcode(on)
            @test relerr(on.u) <= relerr(off.u)
            @test relerr(on.u) < 1.0e-10
        end

        # `lu!` takes no `control`, so the reuse path can only inherit it from the
        # factorization the first solve built. Check it is not silently dropped.
        @testset "setting survives a cached re-solve" begin
            cache = init(
                LinearProblem(copy(A), copy(b)),
                UMFPACKFactorization(max_iterative_refinement_steps = 2)
            )
            solve!(cache)
            @test cache_irstep(cache) == 2

            A2 = copy(A)
            A2.nzval .*= 1.0000001
            cache.A = A2
            sol = solve!(cache)
            @test SciMLBase.successful_retcode(sol)
            @test cache_irstep(cache) == 2
            @test norm(A2 * sol.u - b) < 1.0e-8
        end

        @testset "honored with reuse_symbolic = false" begin
            cache = init(
                LinearProblem(copy(A), copy(b)),
                UMFPACKFactorization(
                    reuse_symbolic = false, max_iterative_refinement_steps = 2
                )
            )
            sol = solve!(cache)
            @test SciMLBase.successful_retcode(sol)
            @test cache_irstep(cache) == 2
        end

        # The control vector is built per solve, so one algorithm's setting must
        # not leak into a cache that asked for a different one.
        @testset "setting does not leak between caches" begin
            refined = init(
                LinearProblem(copy(A), copy(b)),
                UMFPACKFactorization(max_iterative_refinement_steps = 4)
            )
            solve!(refined)
            plain = init(LinearProblem(copy(A), copy(b)), UMFPACKFactorization())
            solve!(plain)
            @test cache_irstep(refined) == 4
            @test cache_irstep(plain) == 0
        end
    end
end
