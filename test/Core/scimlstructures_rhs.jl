using LinearSolve, LinearAlgebra, Test, Random
import SciMLStructures as SS

# A right-hand side that implements the SciMLStructures interface but is not an array
# used to fail in `__init` at `real(eltype(b))`, before any algorithm was reached.
# It is now canonicalized to a flat buffer and repacked. See SciML/LinearSolve.jl#1208.
#
# The type below is the parameter object from the SciMLStructures documentation.
mutable struct SubproblemParameters{P, Q, R}
    p::P
    q::Q
    r::R
end

mutable struct Parameters{P, C}
    subparams::P
    coeffs::C
end

SS.isscimlstructure(::Parameters) = true
SS.ismutablescimlstructure(::Parameters) = true
SS.hasportion(::SS.Tunable, ::Parameters) = true

function SS.canonicalize(::SS.Tunable, p::Parameters)
    buffer = vcat([subpar.p for subpar in p.subparams], vec(p.coeffs))
    repack = let p = p
        newbuffer -> SS.replace(SS.Tunable(), p, newbuffer)
    end
    return buffer, repack, false
end

function SS.replace(::SS.Tunable, p::Parameters, newbuffer)
    subparams = [
        SubproblemParameters(newbuffer[i], s.q, s.r)
            for (i, s) in enumerate(p.subparams)
    ]
    coeffs = reshape(
        view(newbuffer, (length(p.subparams) + 1):length(newbuffer)), size(p.coeffs)
    )
    return Parameters(subparams, coeffs)
end

function SS.replace!(::SS.Tunable, p::Parameters, newbuffer)
    for (s, v) in zip(p.subparams, newbuffer)
        s.p = v
    end
    copyto!(p.coeffs, view(newbuffer, (length(p.subparams) + 1):length(newbuffer)))
    return nothing
end

flatten(x) = first(SS.canonicalize(SS.Tunable(), x))

@testset "SciMLStructures right-hand side (#1208)" begin
    Random.seed!(1208)
    n = 5
    m = 10
    params() = Parameters(
        [SubproblemParameters(0.1i, 0.2i, 0.3i) for i in 1:n],
        cos.([0.1i + 0.33j for i in 1:n, j in 1:m])
    )
    p = params()
    N = length(flatten(p))
    A = rand(N, N) + N * I
    reference = A \ flatten(p)

    @testset "solves and comes back in the caller's type" begin
        for alg in (nothing, LUFactorization(), QRFactorization(), KrylovJL_GMRES())
            sol = alg === nothing ? solve(LinearProblem(A, p)) :
                solve(LinearProblem(A, p), alg)
            @test sol.retcode == LinearSolve.ReturnCode.Success
            @test sol.u isa Parameters
            @test flatten(sol.u) ≈ reference rtol = 1.0e-5
            # The structure is rebuilt, not flattened away.
            @test length(sol.u.subparams) == n
            @test size(sol.u.coeffs) == (n, m)
        end
    end

    # The solution is written into its own container, so solving must not clobber the
    # right-hand side the caller passed in.
    @testset "the right-hand side is not overwritten" begin
        q = params()
        before = copy(flatten(q))
        solve(LinearProblem(A, q), LUFactorization())
        @test flatten(q) == before
    end

    @testset "init and repeated solves" begin
        cache = init(LinearProblem(A, params()), KrylovJL_GMRES())
        @test flatten(solve!(cache).u) ≈ reference rtol = 1.0e-5

        # A new right-hand side has to be re-canonicalized into the flat cache.
        p2 = Parameters(
            [SubproblemParameters(0.5i, 0.2i, 0.3i) for i in 1:n], ones(n, m)
        )
        cache.b = p2
        sol = solve!(cache)
        @test sol.u isa Parameters
        @test flatten(sol.u) ≈ A \ flatten(p2) rtol = 1.0e-5

        # And a new matrix.
        A2 = rand(N, N) + 2N * I
        cache.A = A2
        @test flatten(solve!(cache).u) ≈ A2 \ flatten(p2) rtol = 1.0e-5
    end

    @testset "the cache exposes the flat problem it solves" begin
        cache = init(LinearProblem(A, params()), LUFactorization())
        @test cache.b isa Parameters
        @test cache.u isa Parameters
        # Forwarded from the inner flat cache.
        @test cache.maxiters == N
        @test cache.abstol isa Real
    end

    # Array right-hand sides must not be diverted through this path.
    @testset "ordinary arrays are unaffected" begin
        bvec = rand(N)
        sol = solve(LinearProblem(A, bvec), LUFactorization())
        @test sol.u isa Vector
        @test sol.u ≈ A \ bvec
    end
end
