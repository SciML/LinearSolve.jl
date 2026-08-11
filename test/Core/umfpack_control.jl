using LinearSolve, SparseArrays, LinearAlgebra, Test, Random

# `UMFPACKFactorization` exposes UMFPACK's control vector as a `NamedTuple` of
# named settings. Anything not named keeps whatever SparseArrays defaults it to,
# which notably means iterative refinement stays off (JuliaLang/julia#122) unless
# `irstep` is asked for. See SciML/LinearSolve.jl#383.
@static if Base.USE_GPL_LIBS
    using SparseArrays.UMFPACK: JL_UMFPACK_IRSTEP, JL_UMFPACK_PIVOT_TOLERANCE,
        JL_UMFPACK_PRL, get_umfpack_control

    # The control vector actually handed to UMFPACK for this cache.
    cache_control(cache) = cache.cacheval.control

    Random.seed!(383)
    # Diagonally dominant apart from one tiny pivot, so the solve is accurate
    # enough to be meaningful but ill conditioned enough for refinement to bite.
    n = 60
    A = sprand(n, n, 0.25) + n * I
    A[1, 1] = 1.0e-9
    xtrue = rand(n)
    b = A * xtrue
    relerr(u) = norm(u - xtrue) / norm(xtrue)

    @testset "UMFPACKFactorization control settings (#383)" begin
        @testset "construction" begin
            @test UMFPACKFactorization().control == (;)
            @test UMFPACKFactorization(control = (; irstep = 2)).control ==
                (; irstep = 2)
            # The other keywords keep working alongside it.
            alg = UMFPACKFactorization(
                reuse_symbolic = false, control = (; irstep = 0)
            )
            @test alg.reuse_symbolic == false
            @test alg.control == (; irstep = 0)
        end

        @testset "unknown settings are rejected" begin
            @test_throws ArgumentError UMFPACKFactorization(
                control = (; not_a_setting = 1)
            )
            # A typo of a real name is caught too, rather than silently ignored.
            @test_throws ArgumentError UMFPACKFactorization(control = (; irsteps = 2))
            err = try
                UMFPACKFactorization(control = (; nope = 1))
                nothing
            catch e
                e
            end
            @test occursin("nope", err.msg)
            @test occursin("irstep", err.msg)
        end

        # Every documented name must land on its own entry and disturb no other.
        # Built directly rather than through a solve so the probe value never has
        # to be a legal setting for that particular knob.
        @testset "every documented setting maps to its own entry" begin
            reference = get_umfpack_control(Float64, Int64)
            for setting in LinearSolve.UMFPACK_CONTROL_KEYS
                idx = LinearSolve._UMFPACK_CONTROL_INDEX[setting]
                probe = reference[idx] + 1
                control = LinearSolve._umfpack_control(
                    UMFPACKFactorization(control = NamedTuple{(setting,)}((probe,))),
                    Float64, Int64
                )
                @test control[idx] == probe
                moved = findall(i -> control[i] != reference[i], eachindex(reference))
                @test moved == [idx]
            end
            # And the names cover the whole mapping, so neither list can drift.
            @test Set(LinearSolve.UMFPACK_CONTROL_KEYS) ==
                Set(keys(LinearSolve._UMFPACK_CONTROL_INDEX))
        end

        @testset "the setting reaches UMFPACK" begin
            for steps in (0, 1, 2, 4)
                cache = init(
                    LinearProblem(copy(A), copy(b)),
                    UMFPACKFactorization(control = (; irstep = steps))
                )
                sol = solve!(cache)
                @test SciMLBase.successful_retcode(sol)
                @test cache_control(cache)[JL_UMFPACK_IRSTEP] == steps
            end
        end

        @testset "several settings at once" begin
            cache = init(
                LinearProblem(copy(A), copy(b)),
                UMFPACKFactorization(
                    control = (; irstep = 2, pivot_tolerance = 0.5, prl = 1)
                )
            )
            sol = solve!(cache)
            @test SciMLBase.successful_retcode(sol)
            control = cache_control(cache)
            @test control[JL_UMFPACK_IRSTEP] == 2
            @test control[JL_UMFPACK_PIVOT_TOLERANCE] == 0.5
            @test control[JL_UMFPACK_PRL] == 1
        end

        @testset "unnamed entries keep SparseArrays' defaults" begin
            reference = get_umfpack_control(Float64, Int64)
            cache = init(
                LinearProblem(copy(A), copy(b)),
                UMFPACKFactorization(control = (; irstep = 2))
            )
            solve!(cache)
            control = cache_control(cache)
            # Only the named entry moved; a hand-built vector would have dropped
            # the rest.
            @test control[JL_UMFPACK_IRSTEP] == 2
            for i in eachindex(reference)
                i == JL_UMFPACK_IRSTEP && continue
                @test control[i] == reference[i]
            end
        end

        @testset "default leaves SparseArrays' own default in place" begin
            plain = init(LinearProblem(copy(A), copy(b)), UMFPACKFactorization())
            solve!(plain)
            # SparseArrays disables refinement, so the default must match an
            # explicit 0 rather than pinning a value of its own.
            explicit = init(
                LinearProblem(copy(A), copy(b)),
                UMFPACKFactorization(control = (; irstep = 0))
            )
            solve!(explicit)
            @test cache_control(plain)[JL_UMFPACK_IRSTEP] ==
                cache_control(explicit)[JL_UMFPACK_IRSTEP]
        end

        @testset "refinement improves accuracy" begin
            off = solve(
                LinearProblem(copy(A), copy(b)),
                UMFPACKFactorization(control = (; irstep = 0))
            )
            on = solve(
                LinearProblem(copy(A), copy(b)),
                UMFPACKFactorization(control = (; irstep = 2))
            )
            @test SciMLBase.successful_retcode(off)
            @test SciMLBase.successful_retcode(on)
            @test relerr(on.u) <= relerr(off.u)
            @test relerr(on.u) < 1.0e-10
        end

        # `lu!` takes no `control`, so the reuse path can only inherit it from the
        # factorization the first solve built. Check it is not silently dropped.
        @testset "settings survive a cached re-solve" begin
            cache = init(
                LinearProblem(copy(A), copy(b)),
                UMFPACKFactorization(control = (; irstep = 2))
            )
            solve!(cache)
            @test cache_control(cache)[JL_UMFPACK_IRSTEP] == 2

            A2 = copy(A)
            A2.nzval .*= 1.0000001
            cache.A = A2
            sol = solve!(cache)
            @test SciMLBase.successful_retcode(sol)
            @test cache_control(cache)[JL_UMFPACK_IRSTEP] == 2
            @test norm(A2 * sol.u - b) < 1.0e-8
        end

        @testset "honored with reuse_symbolic = false" begin
            cache = init(
                LinearProblem(copy(A), copy(b)),
                UMFPACKFactorization(
                    reuse_symbolic = false, control = (; irstep = 2)
                )
            )
            sol = solve!(cache)
            @test SciMLBase.successful_retcode(sol)
            @test cache_control(cache)[JL_UMFPACK_IRSTEP] == 2
        end

        # The control vector is built per solve, so one algorithm's settings must
        # not leak into a cache that asked for different ones. The `Float64`/`Int`
        # cacheval is a shared const, which is what makes this worth asserting.
        @testset "settings do not leak between caches" begin
            refined = init(
                LinearProblem(copy(A), copy(b)),
                UMFPACKFactorization(control = (; irstep = 4))
            )
            solve!(refined)
            plain = init(LinearProblem(copy(A), copy(b)), UMFPACKFactorization())
            solve!(plain)
            @test cache_control(refined)[JL_UMFPACK_IRSTEP] == 4
            @test cache_control(plain)[JL_UMFPACK_IRSTEP] == 0
        end
    end
end
