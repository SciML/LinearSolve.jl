using BorderedLinearSolve
using LinearAlgebra, SparseArrays, Random, Test
using LinearSolve: LUFactorization, GenericLUFactorization, QRFactorization, KrylovJL_GMRES,
    LinearAliasSpecifier
using SciMLOperators: FunctionOperator, MatrixOperator
using SciMLBase: successful_retcode

# Reference: assemble the bordered matrix and solve it densely. Everything below is
# checked against this rather than against another code path in the package.
function reference(J, b, c, d, f, g)
    M = [Matrix(J) reshape(b, :, 1); reshape(c, 1, :) fill(d, 1, 1)]
    x = M \ vcat(f, g)
    return x[1:length(f)], x[end]
end

# Reading the last entry of a broken matrix throws `err`, so a refresh that copies it fails partway.
# The matrix is mutable, so a test can change or break one the cache holds and repair it later.
mutable struct BreakableMatrix <: AbstractMatrix{Float64}
    data::Matrix{Float64}
    broken::Bool
    err::Exception
end
BreakableMatrix(data, broken) = BreakableMatrix(data, broken, ErrorException("read the last entry of a broken matrix"))
Base.size(A::BreakableMatrix) = size(A.data)
function Base.getindex(A::BreakableMatrix, i::Int, j::Int)
    A.broken && (i, j) == size(A) && throw(A.err)
    return A.data[i, j]
end

# An exception with an identity of its own, so a test can check that the caller gets the very
# object thrown.
mutable struct Sentinel <: Exception end

# A matrix-free `J` that scales row `i` of a state of any shape by `i`.
struct RowScaling end
Base.:*(::RowScaling, x::AbstractArray) = axes(x, 1) .* x

@testset "BorderedLinearSolve" begin
    Random.seed!(560)

    @testset "bordering matches the assembled system" begin
        for n in (5, 50)
            J = rand(n, n) + n * I
            b = rand(n); c = rand(n); d = 1.5
            f = rand(n); g = 0.7
            vref, σref = reference(J, b, c, d, f, g)

            sol = solve(BorderedLinearProblem(J, b, c, d, f, g), BorderingBLS())
            @test successful_retcode(sol)
            @test sol.v ≈ vref rtol = 1.0e-9
            @test sol.σ ≈ σref rtol = 1.0e-9
        end
    end

    @testset "direct assembly matches too" begin
        n = 20
        J = rand(n, n) + n * I
        b = rand(n); c = rand(n); d = 2.0
        f = rand(n); g = -0.3
        vref, σref = reference(J, b, c, d, f, g)

        sol = solve(BorderedLinearProblem(J, b, c, d, f, g), DirectBLS())
        @test sol.v ≈ vref rtol = 1.0e-9
        @test sol.σ ≈ σref rtol = 1.0e-9
    end

    @testset "the inner cache is reused across the two solves" begin
        # This is the reason the package exists, so it is asserted rather than assumed:
        # after the first inner solve the cache is no longer fresh, meaning the second
        # solve reuses the factorization instead of rebuilding it.
        n = 30
        J = rand(n, n) + n * I
        prob = BorderedLinearProblem(J, rand(n), rand(n), 1.0, rand(n), 0.5)
        cache = init(prob, BorderingBLS(LUFactorization()))
        @test cache.inner.isfresh
        solve!(cache)
        @test !cache.inner.isfresh
    end

    @testset "reinit! refactors for a new J and reuses the factorization for a new f and g" begin
        # A cache cannot see a `J` changed in place, so `reinit!` has to say so. Without
        # it the next solve ran against the old factorization and returned a wrong answer
        # with a Success retcode.
        n = 12
        for alg in (BorderingBLS(LUFactorization()), DirectBLS(), DirectBLS(LUFactorization()))
            J = rand(n, n) + n * I
            b = rand(n); c = rand(n); d = 1.5; f = rand(n); g = 0.7
            cache = init(BorderedLinearProblem(J, b, c, d, f, g), alg)
            solve!(cache)

            f2, g2 = rand(n), -0.2
            reinit!(cache; f = f2, g = g2)
            @test !cache.inner.isfresh
            sol = solve!(cache)
            vref, σref = reference(J, b, c, d, f2, g2)
            @test sol.v ≈ vref rtol = 1.0e-9
            @test sol.σ ≈ σref rtol = 1.0e-9

            view(J, diagind(J)) .+= 3
            reinit!(cache; J)
            @test cache.inner.isfresh
            sol = solve!(cache)
            vref, σref = reference(J, b, c, d, f2, g2)
            @test sol.v ≈ vref rtol = 1.0e-9
            @test sol.σ ≈ σref rtol = 1.0e-9

            # The border sits in DirectBLS's assembled matrix, and outside J for bordering.
            b3 = rand(n); c3 = rand(n); d3 = 0.9
            reinit!(cache; b = b3, c = c3, d = d3)
            @test cache.inner.isfresh == (alg isa DirectBLS)
            sol = solve!(cache)
            vref, σref = reference(J, b3, c3, d3, f2, g2)
            @test sol.v ≈ vref rtol = 1.0e-9
            @test sol.σ ≈ σref rtol = 1.0e-9

            J4 = J + 2I
            cache.prob = BorderedLinearProblem(J4, b, c, d, f, g)
            @test cache.inner.isfresh
            sol = solve!(cache)
            vref, σref = reference(J4, b, c, d, f, g)
            @test sol.v ≈ vref rtol = 1.0e-9
            @test sol.σ ≈ σref rtol = 1.0e-9
        end
    end

    @testset "DirectBLS reinit! with a wide border" begin
        n, m = 8, 2
        J = rand(n, n) + n * I
        bt = ntuple(_ -> rand(n), m)
        ct = ntuple(_ -> rand(n), m)
        d = rand(m, m)
        cache = init(BorderedLinearProblem(J, bt, ct, d, rand(n), rand(m)), DirectBLS())
        solve!(cache)
        f2, g2 = rand(n), rand(m)
        reinit!(cache; f = f2, g = g2)
        @test !cache.inner.isfresh
        sol = solve!(cache)
        xref = [J reduce(hcat, collect(bt)); reduce(hcat, collect(ct))' d] \ vcat(f2, g2)
        @test sol.v ≈ xref[1:n] rtol = 1.0e-9
        @test sol.σ ≈ xref[(n + 1):end] rtol = 1.0e-9
    end

    @testset "a sparse J works and keeps J sparse" begin
        n = 60
        J = spdiagm(-1 => -ones(n - 1), 0 => 4ones(n), 1 => -ones(n - 1))
        b = rand(n); c = rand(n); d = 1.0
        f = rand(n); g = 0.25
        vref, σref = reference(J, b, c, d, f, g)

        sol = solve(BorderedLinearProblem(J, b, c, d, f, g), BorderingBLS())
        @test sol.v ≈ vref rtol = 1.0e-8
        @test sol.σ ≈ σref rtol = 1.0e-8
    end

    @testset "an iterative inner solver works" begin
        n = 40
        J = rand(n, n) + n * I
        b = rand(n); c = rand(n); d = 1.0
        f = rand(n); g = 0.1
        vref, σref = reference(J, b, c, d, f, g)

        sol = solve(
            BorderedLinearProblem(J, b, c, d, f, g),
            BorderingBLS(KrylovJL_GMRES())
        )
        @test sol.v ≈ vref rtol = 1.0e-6
        @test sol.σ ≈ σref rtol = 1.0e-6
    end

    @testset "BEC+k corrects a nearly singular J" begin
        # Block elimination loses accuracy as J approaches singular, which is exactly
        # where continuation spends its time. The correction sweeps should recover it.
        n = 30
        U, _ = qr(rand(n, n))
        svals = vcat(1.0e-11, range(1.0, 2.0; length = n - 1))
        J = Matrix(U) * Diagonal(svals) * Matrix(U)'
        b = rand(n); c = rand(n); d = 1.0
        f = rand(n); g = 0.4
        vref, σref = reference(J, b, c, d, f, g)

        plain = solve(BorderedLinearProblem(J, b, c, d, f, g), BorderingBLS(k = 0))
        corrected = solve(BorderedLinearProblem(J, b, c, d, f, g), BorderingBLS(k = 3))

        err(s) = norm(s.v - vref) / norm(vref) + abs(s.σ - σref) / abs(σref)
        println("  nearly singular: k=0 err=", err(plain), "  k=3 err=", err(corrected))
        @test err(corrected) <= err(plain)
    end

    @testset "a wide border" begin
        n, m = 8, 2
        J = rand(n, n) + n * I
        # A wide border is a tuple of states. It is not inferred from an array's
        # shape, because a state can itself be a matrix.
        bt = ntuple(_ -> rand(n), m)
        ct = ntuple(_ -> rand(n), m)
        d = rand(m, m)
        f = rand(n); g = rand(m)
        M = [reduce(hcat, collect(bt)); ]
        M = [J reduce(hcat, collect(bt)); reduce(hcat, collect(ct))' d]
        xref = M \ vcat(f, g)
        prob = BorderedLinearProblem(J, bt, ct, d, f, g)

        # DirectBLS assembles the whole thing, so a wide border is no different.
        sol = solve(prob, DirectBLS())
        @test sol.v ≈ xref[1:n] rtol = 1.0e-9
        @test sol.σ ≈ xref[(n + 1):end] rtol = 1.0e-9

        # Bordering handles it too: m inner solves for the border block, then a
        # small m x m solve. Checked against the same assembled reference.
        solb = solve(prob, BorderingBLS())
        @test successful_retcode(solb)
        @test solb.v ≈ xref[1:n] rtol = 1.0e-9
        @test solb.σ ≈ xref[(n + 1):end] rtol = 1.0e-9

        # And the correction sweeps work for a wide border as well.
        solk = solve(prob, BorderingBLS(k = 2))
        @test solk.v ≈ xref[1:n] rtol = 1.0e-9
        @test solk.σ ≈ xref[(n + 1):end] rtol = 1.0e-9
    end

    @testset "a border may be one state, a tuple of states or a vector of states" begin
        # At width one the correction sweeps took `c` itself for the state, so a tuple or a
        # vector holding one state threw a `DimensionMismatch` once `k` was above zero.
        n = 6
        J = rand(n, n) + n * I; f = rand(n)
        algs = (BorderingBLS(), BorderingBLS(; k = 2), DirectBLS())
        for m in (1, 2), form in (:state, :tuple, :vector), alg in algs
            form === :state && m > 1 && continue
            bs = [rand(n) for _ in 1:m]; cs = [rand(n) for _ in 1:m]
            border(x) = form === :state ? only(x) : form === :tuple ? Tuple(x) : x
            d = m == 1 ? 1.5 : rand(m, m) + m * I; g = m == 1 ? 0.7 : rand(m)
            sol = solve(BorderedLinearProblem(J, border(bs), border(cs), d, f, g), alg)
            x = [J reduce(hcat, bs); reduce(hcat, cs)' d] \ vcat(f, g)
            @test successful_retcode(sol)
            @test sol.v ≈ x[1:n] rtol = 1.0e-9
            @test sol.σ ≈ (m == 1 ? x[end] : x[(n + 1):end]) rtol = 1.0e-9
        end
    end

    @testset "every inner algorithm gives the right answer" begin
        # The GPU run found LUFactorization returning a wrong answer with a Success
        # retcode, which this suite could not have caught: the only test using an
        # explicit inner algorithm checked cache state, never correctness.
        n = 40
        J = rand(n, n) + n * I
        b = rand(n); c = rand(n); d = 1.7; f = rand(n); g = 0.35
        vref, σref = reference(J, b, c, d, f, g)

        for (name, inner) in (
                ("default", nothing),
                ("LUFactorization", LUFactorization()),
                ("KrylovJL_GMRES", KrylovJL_GMRES()),
            )
            for k in (0, 2)
                sol = solve(
                    BorderedLinearProblem(J, b, c, d, f, g),
                    BorderingBLS(inner = inner, k = k)
                )
                @test successful_retcode(sol)
                @test sol.v ≈ vref rtol = 1.0e-6
                @test sol.σ ≈ σref rtol = 1.0e-6
            end
        end
    end

    @testset "the caller's arrays are not mutated" begin
        # A solve that overwrites `b` in place would destroy the problem the user
        # handed us, and corrupt the BEC+k residual, which re-reads it.
        n = 30
        J = rand(n, n) + n * I
        b = rand(n); c = rand(n); d = 1.0; f = rand(n); g = 0.5
        J0, b0, c0, f0 = copy(J), copy(b), copy(c), copy(f)

        for inner in (nothing, LUFactorization(), KrylovJL_GMRES())
            solve(BorderedLinearProblem(J, b, c, d, f, g), BorderingBLS(inner = inner, k = 2))
            @test b == b0
            @test f == f0
            @test c == c0
            @test J == J0
        end
    end

    @testset "a refactor leaves the caller's J intact" begin
        # Most dense inner factorizations overwrite `A` on a refactor. Handing them the
        # caller's `J` destroyed it, and the BEC+k residual then re-read the factors.
        n = 30
        for inner in (nothing, LUFactorization(), GenericLUFactorization(), QRFactorization()), update in (:reinit, :prob)
            J = rand(n, n) + n * I
            b = rand(n); c = rand(n); d = 1.0; f = rand(n); g = 0.5
            cache = init(BorderedLinearProblem(J, b, c, d, f, g), BorderingBLS(; inner, k = 2))
            solve!(cache)
            view(J, diagind(J)) .+= 3
            J0 = copy(J)
            update === :reinit ? reinit!(cache; J) : (cache.prob = BorderedLinearProblem(J, b, c, d, f, g))
            sol = solve!(cache)
            @test J == J0
            vref, σref = reference(J0, b, c, d, f, g)
            @test sol.v ≈ vref rtol = 1.0e-9
            @test sol.σ ≈ σref rtol = 1.0e-9
        end
    end

    @testset "a failed update leaves the cache solving the old problem" begin
        # The new problem was stored before the refresh ran. When the refresh threw, the next
        # solve! paired it with the old factorization: a wrong answer with a Success retcode.
        # After its first refactor DirectBLS factors its matrix in place, so a refresh that
        # fails partway has also overwritten part of the old factors.
        n = 6
        inners = (LUFactorization(), GenericLUFactorization(), QRFactorization())
        algs = Iterators.flatten((BorderingBLS(; inner, k = 0), BorderingBLS(; inner, k = 2), DirectBLS(; inner)) for inner in inners)
        for alg in algs, update in (:reinit, :prob)
            J = rand(n, n) + n * I
            b = rand(n); c = rand(n); d = 1.5; f = rand(n); g = 0.7
            cache = init(BorderedLinearProblem(BreakableMatrix(J, false), b, c, d, f, g), alg)
            solve!(cache)
            reinit!(cache; J = BreakableMatrix(J, false))
            solve!(cache)
            broken, f2 = BreakableMatrix(J + I, true), rand(n)
            fail!() = update === :reinit ? reinit!(cache; J = broken, f = f2) :
                (cache.prob = BorderedLinearProblem(broken, b, c, d, f2, g))
            @test_throws ErrorException fail!()
            @test !cache.prob.J.broken
            sol = solve!(cache)
            vref, σref = reference(J, b, c, d, f, g)
            @test successful_retcode(sol)
            @test sol.v ≈ vref rtol = 1.0e-9
            @test sol.σ ≈ σref rtol = 1.0e-9
        end
    end

    @testset "a failed update rethrows the exception its refresh raised" begin
        # The update restores the old problem and rethrows. The caller must get the very object
        # the refresh threw, not one from reading the old `J` or a wrapper around it.
        n = 6
        inners = (LUFactorization(), GenericLUFactorization(), QRFactorization())
        algs = Iterators.flatten((BorderingBLS(; inner, k = 0), BorderingBLS(; inner, k = 2), DirectBLS(; inner)) for inner in inners)
        for alg in algs, update in (:reinit, :prob)
            J = rand(n, n) + n * I
            b = rand(n); c = rand(n); d = 1.5; f = rand(n); g = 0.7
            old = BreakableMatrix(J, false)
            cache = init(BorderedLinearProblem(old, b, c, d, f, g), alg)
            solve!(cache)
            # Reading the old `J` now throws an `ErrorException`, and reading the new one a `Sentinel`.
            old.broken = true
            broken = BreakableMatrix(J + I, true, Sentinel())
            thrown = try
                update === :reinit ? reinit!(cache; J = broken) :
                    (cache.prob = BorderedLinearProblem(broken, b, c, d, f, g))
                nothing
            catch err
                err
            end
            @test thrown === broken.err
        end
    end

    @testset "a failed update leaves the cache to refactor from J as it is" begin
        # After `J` changed in place, the refresh and then the restore read it, and both threw
        # before the inner cache took it. The next solve! reused the factors of `J` as it was
        # before the change: a wrong answer with a Success retcode.
        n = 6
        inners = (LUFactorization(), GenericLUFactorization(), QRFactorization())
        algs = Iterators.flatten((BorderingBLS(; inner, k = 0), BorderingBLS(; inner, k = 2), DirectBLS(; inner)) for inner in inners)
        for alg in algs, update in (:reinit, :prob)
            J = BreakableMatrix(rand(n, n) + n * I, false)
            b = rand(n); c = rand(n); d = 1.5; f = rand(n); g = 0.7
            cache = init(BorderedLinearProblem(J, b, c, d, f, g), alg)
            solve!(cache)
            view(J.data, diagind(J.data)) .+= 1
            J.broken = true
            @test_throws ErrorException update === :reinit ? reinit!(cache; J) :
                (cache.prob = BorderedLinearProblem(J, b, c, d, f, g))
            # Until `J` can be read again, a solve throws, and so does an update that keeps the factorization.
            @test_throws ErrorException solve!(cache)
            @test_throws ErrorException reinit!(cache; g = -0.3)
            J.broken = false
            sol = solve!(cache)
            vref, σref = reference(J.data, b, c, d, f, g)
            @test successful_retcode(sol)
            @test sol.v ≈ vref rtol = 1.0e-9
            @test sol.σ ≈ σref rtol = 1.0e-9
            # Refactored, the cache keeps its factorization for a new right-hand side again.
            reinit!(cache; f = rand(n))
            @test !cache.inner.isfresh
        end
    end

    @testset "a rejected update that passes J leaves the cache to refactor" begin
        # After `J` changed in place, an update that passed it but was rejected, for a part of
        # another type or size, threw before marking the factors stale. The next solve! reused
        # the factors of `J` from before the change: a wrong answer with a Success retcode.
        n = 6
        inners = (LUFactorization(), GenericLUFactorization(), QRFactorization())
        algs = Iterators.flatten((BorderingBLS(; inner, k = 0), BorderingBLS(; inner, k = 2), DirectBLS(; inner)) for inner in inners)
        for alg in algs, rejected in (:type, :size, :prob)
            J = rand(n, n) + n * I
            b = rand(n); c = rand(n); d = 1.5; f = rand(n); g = 0.7
            cache = init(BorderedLinearProblem(J, b, c, d, f, g), alg)
            solve!(cache)
            view(J, diagind(J)) .+= 1
            if rejected === :type
                @test_throws ArgumentError reinit!(cache; J, d = 2)
            elseif rejected === :size
                @test_throws DimensionMismatch reinit!(cache; J, f = rand(n + 1))
            else
                @test_throws ArgumentError cache.prob = BorderedLinearProblem(J, b, c, 2, f, g)
            end
            sol = solve!(cache)
            vref, σref = reference(J, b, c, d, f, g)
            @test successful_retcode(sol)
            @test sol.v ≈ vref rtol = 1.0e-9
            @test sol.σ ≈ σref rtol = 1.0e-9
        end
    end

    @testset "a failed update whose old J cannot be read leaves no stale factors" begin
        # After its first refactor DirectBLS factors its assembled matrix in place, and an
        # assembly that throws partway leaves half-written factors. If the old `J` threw too,
        # they stayed marked current: a wrong answer with a Success retcode.
        n = 6
        inners = (LUFactorization(), GenericLUFactorization(), QRFactorization())
        algs = Iterators.flatten((BorderingBLS(; inner, k = 0), BorderingBLS(; inner, k = 2), DirectBLS(; inner)) for inner in inners)
        aliasing = (NamedTuple(), (; alias = LinearAliasSpecifier(alias_A = true)))
        for alg in algs, kwargs in aliasing, update in (:reinit, :prob), next in (:solve, :rhs)
            # Aliasing would hand an in-place factorization the caller's read-only `J`.
            alg isa BorderingBLS && !isempty(kwargs) && continue
            J = rand(n, n) + n * I
            b = rand(n); c = rand(n); d = 1.5; f = rand(n); g = 0.7
            old = BreakableMatrix(J, false)
            cache = init(BorderedLinearProblem(old, b, c, d, f, g), alg; kwargs...)
            solve!(cache)
            reinit!(cache; J = old)
            solve!(cache)
            old.broken = true
            broken = BreakableMatrix(J + I, true)
            @test_throws ErrorException update === :reinit ? reinit!(cache; J = broken, f = rand(n)) :
                (cache.prob = BorderedLinearProblem(broken, b, c, d, rand(n), g))
            @test cache.prob.J === old
            # Until the old `J` can be read again, a solve throws, and so does an update that keeps the factorization.
            @test_throws ErrorException solve!(cache)
            @test_throws ErrorException reinit!(cache; g = -0.3)
            old.broken = false
            if next === :rhs
                g = -0.3
                reinit!(cache; g)
            end
            sol = solve!(cache)
            vref, σref = reference(J, b, c, d, f, g)
            @test successful_retcode(sol)
            @test sol.v ≈ vref rtol = 1.0e-9
            @test sol.σ ≈ σref rtol = 1.0e-9
            # Refactored, the cache keeps its factorization for a new right-hand side again.
            reinit!(cache; f = rand(n))
            @test !cache.inner.isfresh
        end
    end

    @testset "an update may change the state size or the border width" begin
        # The caches kept their buffers for the old sizes. Bordering then threw at the next
        # solve!, and DirectBLS rewrote only part of its old matrix and returned a wrong answer
        # with a Success retcode.
        inners = (
            (nothing, 1.0e-9), (LUFactorization(), 1.0e-9), (GenericLUFactorization(), 1.0e-9),
            (QRFactorization(), 1.0e-9), (KrylovJL_GMRES(), 1.0e-6),
        )
        for (inner, rtol) in inners, alg in (BorderingBLS(; inner, k = 2), DirectBLS(; inner)), update in (:reinit, :prob)
            for (n1, n2) in ((6, 3), (3, 6))
                J = rand(n1, n1) + n1 * I
                prob = BorderedLinearProblem(J, rand(n1), rand(n1), 1.5, rand(n1), 0.7)
                cache = init(prob, alg; reltol = 1.0e-10)
                solve!(cache)
                J2 = rand(n2, n2) + n2 * I
                b = rand(n2); c = rand(n2); d = 0.9; f = rand(n2); g = -0.2
                update === :reinit ? reinit!(cache; J = J2, b, c, d, f, g) :
                    (cache.prob = BorderedLinearProblem(J2, b, c, d, f, g))
                # The new inner cache is built with the keywords given to `init`.
                @test cache.inner.reltol == 1.0e-10
                sol = solve!(cache)
                vref, σref = reference(J2, b, c, d, f, g)
                @test successful_retcode(sol)
                @test sol.v ≈ vref rtol = rtol
                @test sol.σ ≈ σref rtol = rtol
            end

            n = 5
            J = rand(n, n) + n * I; f = rand(n)
            border(m) = [rand(n) for _ in 1:m]
            for (m1, m2) in ((2, 3), (3, 2))
                cache = init(BorderedLinearProblem(J, border(m1), border(m1), rand(m1, m1), f, rand(m1)), alg)
                solve!(cache)
                b, c, d, g = border(m2), border(m2), rand(m2, m2), rand(m2)
                update === :reinit ? reinit!(cache; b, c, d, g) :
                    (cache.prob = BorderedLinearProblem(J, b, c, d, f, g))
                sol = solve!(cache)
                x = [J reduce(hcat, b); reduce(hcat, c)' d] \ vcat(f, g)
                @test successful_retcode(sol)
                @test sol.v ≈ x[1:n] rtol = rtol
                @test sol.σ ≈ x[(n + 1):end] rtol = rtol
            end
        end
    end

    @testset "an update may reshape the state" begin
        # A matrix-free `J` acts on states of one shape. Bordering kept its operator for the old
        # shape when the length stayed the same: a wrong answer with a Success retcode.
        algs = (
            BorderingBLS(), BorderingBLS(; k = 2),
            BorderingBLS(; inner = KrylovJL_GMRES()), BorderingBLS(; inner = KrylovJL_GMRES(), k = 2),
        )
        for alg in algs, update in (:reinit, :prob)
            cache = init(BorderedLinearProblem(RowScaling(), rand(2, 3), rand(2, 3), 1.5, rand(2, 3), 0.7), alg)
            solve!(cache)
            b = rand(3, 2); c = rand(3, 2); f = rand(3, 2)
            update === :reinit ? reinit!(cache; b, c, f) :
                (cache.prob = BorderedLinearProblem(RowScaling(), b, c, 1.5, f, 0.7))
            sol = solve!(cache)
            vref, σref = reference(kron(I(2), Diagonal(1.0:3.0)), vec(b), vec(c), 1.5, vec(f), 0.7)
            @test successful_retcode(sol)
            @test size(sol.v) == (3, 2)
            @test vec(sol.v) ≈ vref rtol = 1.0e-6
            @test sol.σ ≈ σref rtol = 1.0e-6
        end
        # Assembly applies a matrix `J` to the flattened state, whatever its shape.
        for alg in (DirectBLS(), DirectBLS(LUFactorization())), update in (:reinit, :prob)
            J = rand(6, 6) + 6I
            cache = init(BorderedLinearProblem(J, rand(2, 3), rand(2, 3), 1.5, rand(2, 3), 0.7), alg)
            solve!(cache)
            b = rand(3, 2); c = rand(3, 2); f = rand(3, 2)
            update === :reinit ? reinit!(cache; b, c, f) :
                (cache.prob = BorderedLinearProblem(J, b, c, 1.5, f, 0.7))
            sol = solve!(cache)
            vref, σref = reference(J, vec(b), vec(c), 1.5, vec(f), 0.7)
            @test successful_retcode(sol)
            @test size(sol.v) == (3, 2)
            @test vec(sol.v) ≈ vref rtol = 1.0e-9
            @test sol.σ ≈ σref rtol = 1.0e-9
        end
    end

    @testset "an update cannot change the type of a part" begin
        # A cache's fields are typed at `init`. A tuple border of another length, or a wide
        # border for a single one, threw a `TypeError` from `setfield!` or a `MethodError`.
        n = 5
        J = rand(n, n) + n * I; f = rand(n)
        border(m) = m == 1 ?
            (b = rand(n), c = rand(n), d = 1.5, g = 0.7) :
            (b = ntuple(_ -> rand(n), m), c = ntuple(_ -> rand(n), m), d = rand(m, m) + m * I, g = rand(m))
        algs = (BorderingBLS(LUFactorization()), BorderingBLS(; k = 2), DirectBLS())
        for alg in algs, (m1, m2) in ((2, 3), (1, 2), (2, 1)), update in (:reinit, :prob)
            old, new = border(m1), border(m2)
            cache = init(BorderedLinearProblem(J, old.b, old.c, old.d, f, old.g), alg)
            sol = solve!(cache)
            # A rejected update that would have refactored leaves the cache stale; see the testset above.
            names = filter(!=(:stale), fieldnames(typeof(cache)))
            fields = map(name -> getfield(cache, name), names)
            @test_throws ArgumentError update === :reinit ? reinit!(cache; new.b, new.c, new.d, new.g) :
                (cache.prob = BorderedLinearProblem(J, new.b, new.c, new.d, f, new.g))
            # Nothing else changed: the cache holds the same objects and solves the old problem as before.
            @test all(map(===, fields, map(name -> getfield(cache, name), names)))
            alg isa BorderingBLS && @test !cache.inner.isfresh
            sol2 = solve!(cache)
            @test sol2.v == sol.v
            @test sol2.σ == sol.σ
        end
        # The error names the parts whose type changed and points to `init`.
        cache = init(BorderedLinearProblem(J, rand(n), rand(n), 1.5, f, 0.7), BorderingBLS())
        @test_throws ["d from Float64 to Int64", "call init"] reinit!(cache; d = 2)
    end

    @testset "a state that is not a vector" begin
        # A 2D problem keeps its state as a matrix. Requiring a vector would push
        # `vec`/`reshape` onto the caller, so the state shape is preserved end to end
        # and only the inner solve sees a flattened form.
        N = 4
        a = 3.0
        f2 = rand(N, N); b2 = rand(N, N); c2 = rand(N, N); d2 = 1.7; g2 = 0.3

        apply(v, u, p, t) = a .* v
        apply(w, v, u, p, t) = (w .= a .* v; w)
        J2 = FunctionOperator(apply, zeros(N, N), zeros(N, N); islinear = true)

        prob2 = BorderedLinearProblem(J2, b2, c2, d2, f2, g2)
        sol2 = solve(prob2, BorderingBLS(KrylovJL_GMRES()))

        @test successful_retcode(sol2)
        # The state comes back in its own shape, not flattened.
        @test size(sol2.v) == size(f2)
        @test sol2.σ isa Number

        nn = N * N
        M2 = [a * Matrix(I, nn, nn) vec(b2); vec(c2)' fill(d2, 1, 1)]
        xref2 = M2 \ vcat(vec(f2), g2)
        @test vec(sol2.v) ≈ xref2[1:nn] rtol = 1.0e-8
        @test sol2.σ ≈ xref2[end] rtol = 1.0e-8
    end

    @testset "shape errors are caught" begin
        # The state fixes n, and b fixes the border width m; every other part must match them.
        # DirectBLS copied a misshapen J into its matrix entry by entry, then returned a wrong
        # answer with a Success retcode.
        n = 5
        J = rand(n, n); x = rand(n); wide = (rand(n), rand(n))
        for parts in (
                (rand(1, 4), rand(2), rand(2), 1.0, rand(2), 1.0),
                (rand(2, 2), rand(3), rand(3), 1.0, rand(3), 1.0),
                (rand(n, n + 1), x, x, 1.0, x, 1.0),
                (MatrixOperator(rand(n + 1, n + 1)), x, x, 1.0, x, 1.0),
                (J, rand(n + 1), x, 1.0, x, 1.0),
                (J, x, rand(n + 1), 1.0, x, 1.0),
                (J, x, x, 1.0, rand(n + 1), 1.0),
                (J, wide, x, 1.0, x, 1.0),
                (J, x, x, fill(1.0, 1, 1), x, 1.0),
                (J, x, x, 1.0, x, [1.0]),
                (J, wide, wide, rand(3, 3), x, rand(2)),
                (J, wide, wide, 1.0, x, rand(2)),
                (J, wide, wide, rand(2, 2), x, rand(3)),
                (J, wide, wide, rand(2, 2), x, 1.0),
            )
            @test_throws DimensionMismatch BorderedLinearProblem(parts...)
        end
        # A tuple `g` has the right length but is not the vector bordering subtracts from.
        @test_throws "expected a vector of length 2" BorderedLinearProblem(J, wide, wide, rand(2, 2), x, (1.0, 2.0))

        # On a state that is not a vector, `J` acts on the state's own shape and is left to the
        # algorithm. Bordering applies this 2 x 2 `J` to each column of the 2 x 2 state, while
        # DirectBLS needs the 4 x 4 matrix of the whole map.
        prob = BorderedLinearProblem(rand(2, 2) + 2I, rand(2, 2), rand(2, 2), 1.0, rand(2, 2), 1.0)
        @test successful_retcode(solve(prob, BorderingBLS(KrylovJL_GMRES())))
        @test_throws DimensionMismatch init(prob, DirectBLS())
    end
end
