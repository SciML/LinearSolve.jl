using BorderedLinearSolve
using LinearAlgebra, SparseArrays, Random, Test
using LinearSolve: AbstractFactorization, LUFactorization, GenericLUFactorization, QRFactorization,
    KrylovJL_GMRES, LinearAliasSpecifier, LinearSolveFunction, NormalCholeskyFactorization
using SciMLOperators: FunctionOperator, MatrixOperator
using SciMLBase: successful_retcode

# The assembled bordered matrix. A wide border is a tuple of states, with `d` an m x m matrix.
function assembled(J, b, c, d)
    columns(x) = reduce(hcat, [vec(xi) for xi in (x isa Tuple ? x : (x,))])
    return [Matrix(J) columns(b); columns(c)' (d isa Number ? fill(d, 1, 1) : d)]
end

# Reference: assemble the bordered matrix and solve it densely. Everything below is
# checked against this rather than against another code path in the package. With a wide
# border, `g` is a vector.
function reference(J, b, c, d, f, g)
    x = assembled(J, b, c, d) \ vcat(vec(f), g)
    n = length(f)
    return x[1:n], g isa Number ? x[end] : x[(n + 1):end]
end

# Test `sol` against the dense reference for the problem it solved.
function test_reference(sol, J, b, c, d, f, g; rtol)
    vref, σref = reference(J, b, c, d, f, g)
    @test successful_retcode(sol)
    @test vec(sol.v) ≈ vref rtol = rtol
    @test sol.σ ≈ σref rtol = rtol
    return nothing
end

# Every inner algorithm the correctness tests run, with the tolerance its answers meet.
# Each such testset loops over all of them: separate lists per testset once missed a bug
# that only the algorithms left out had.
const INNERS = (
    (nothing, 1.0e-9),
    (LUFactorization(), 1.0e-9),
    (GenericLUFactorization(), 1.0e-9),
    (QRFactorization(), 1.0e-9),
    (KrylovJL_GMRES(), 1.0e-6),
)

# A backend that solves and then overwrites its right-hand side, as some in-place solvers do,
# counting its calls in `OVERWRITES`. It stays out of `INNERS`: a `LinearSolveFunction` never
# clears `isfresh`, which the `reinit!` tests check.
const OVERWRITES = Ref(0)
const OVERWRITING = LinearSolveFunction(
    function (A, b, u, p, isfresh, Pl, Pr, cacheval; kwargs...)
        OVERWRITES[] += 1
        u .= A \ b
        fill!(b, NaN)
        return u
    end
)

# Bordering without and with correction sweeps (which re-read the problem), and assembly.
bordered_algorithms(inner; det_sign = false) =
    (BorderingBLS(; inner, det_sign), BorderingBLS(; inner, k = 2, det_sign), DirectBLS(; inner, det_sign))

# Run `step()`, test that it leaves every array in `inputs` as it found it, and return its
# result. A solve must not overwrite the caller's problem, which BEC+k also re-reads.
function leaving_intact(step, inputs...)
    snapshots = map(copy, inputs)
    result = step()
    for (input, snapshot) in zip(inputs, snapshots)
        @test input == snapshot
    end
    return result
end

# The parts of a well-posed problem with state length `n` and border width `m`.
function random_parts(n, m)
    border() = m == 1 ? rand(n) : ntuple(_ -> rand(n), m)
    return (;
        J = rand(n, n) + n * I, b = border(), c = border(),
        d = m == 1 ? 1.5 : rand(m, m) + m * I, f = rand(n), g = m == 1 ? 0.7 : rand(m),
    )
end

# Change a part in place: the diagonal of a matrix, which keeps it well conditioned, every
# entry of a vector, and the first state of a wide border.
change!(x::AbstractMatrix) = (view(x, diagind(x)) .+= 3; x)
change!(x::AbstractVector) = (x .= rand(length(x)); x)
change!(x::Tuple) = (change!(first(x)); x)

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

    @testset "every algorithm matches the assembled system" begin
        for n in (5, 50), (inner, rtol) in INNERS, alg in bordered_algorithms(inner)
            J = rand(n, n) + n * I
            b = rand(n); c = rand(n); d = 1.5; f = rand(n); g = 0.7
            sol = leaving_intact(J, b, c, f) do
                solve(BorderedLinearProblem(J, b, c, d, f, g), alg)
            end
            test_reference(sol, J, b, c, d, f, g; rtol)
        end
    end

    @testset "a backend that overwrites its right-hand side spoils nothing" begin
        # Bordering hands the inner solver `f` and each border part as right-hand sides, and
        # each correction sweep hands it the residuals, re-read from `f`. Each is copied in
        # first, or this backend would overwrite the caller's arrays and the sweeps would use
        # the garbage. A zero tolerance makes BEC+k run every sweep.
        states(x) = x isa Tuple ? x : (x,)
        n = 6
        algs = (
            BorderingBLS(; inner = OVERWRITING), BorderingBLS(; inner = OVERWRITING, k = 2, tol = 0.0),
            DirectBLS(; inner = OVERWRITING),
        )
        for alg in algs, m in (1, 2)
            parts = random_parts(n, m)
            cache = init(BorderedLinearProblem(parts...), alg)
            for _ in 1:2
                calls = OVERWRITES[]
                sol = leaving_intact(() -> solve!(cache), parts.J, states(parts.b)..., states(parts.c)..., parts.f)
                # Bordering solves for `f` and each border part in the first pass and in each sweep.
                @test OVERWRITES[] - calls == (alg isa DirectBLS ? 1 : (1 + m) * (1 + alg.k))
                test_reference(sol, parts...; rtol = 1.0e-9)
                # A new right-hand side goes through `reinit!` and the same copies.
                parts = merge(parts, (; f = rand(n)))
                reinit!(cache; parts.f)
            end
        end
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
        leaving_intact(() -> solve!(cache), J)
        @test !cache.inner.isfresh
    end

    @testset "reinit! refactors for exactly the parts in the factorization" begin
        # A cache cannot see a part changed in place, so `reinit!` has to say so. Without it
        # the next solve ran against the old factorization and returned a wrong answer with a
        # Success retcode. Each part changes alone, so a cache that misses any one fails here.
        # The border sits in DirectBLS's assembled matrix, and outside `J` for bordering. A
        # wide border makes `d` and `g` arrays, which can change in place too.
        n = 12
        for (inner, rtol) in INNERS, alg in bordered_algorithms(inner), m in (1, 2)
            refactoring = alg isa DirectBLS ? (:J, :b, :c, :d) : (:J,)
            for part in (:J, :b, :c, :d, :f, :g), in_place in (false, true)
                in_place && m == 1 && part in (:d, :g) && continue
                parts = random_parts(n, m)
                cache = init(BorderedLinearProblem(parts...), alg)
                leaving_intact(() -> solve!(cache), parts.J, parts.f)

                changed = NamedTuple{(part,)}((in_place ? change!(parts[part]) : random_parts(n, m)[part],))
                reinit!(cache; changed...)
                @test cache.inner.isfresh == (part in refactoring)
                parts = merge(parts, changed)
                sol = leaving_intact(() -> solve!(cache), parts.J, parts.f)
                test_reference(sol, parts...; rtol)
            end

            parts = random_parts(n, m)
            cache = init(BorderedLinearProblem(parts...), alg)
            solve!(cache)
            parts = random_parts(n, m)
            cache.prob = BorderedLinearProblem(parts...)
            @test cache.inner.isfresh
            test_reference(solve!(cache), parts...; rtol)
        end
    end

    @testset "a sparse J works and keeps J sparse" begin
        n = 60
        J = spdiagm(-1 => -ones(n - 1), 0 => 4ones(n), 1 => -ones(n - 1))
        b = rand(n); c = rand(n); d = 1.0
        f = rand(n); g = 0.25
        sol = leaving_intact(J, b, c, f) do
            solve(BorderedLinearProblem(J, b, c, d, f, g), BorderingBLS())
        end
        test_reference(sol, J, b, c, d, f, g; rtol = 1.0e-8)
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

        plain, corrected = leaving_intact(J, b, c, f) do
            map(k -> solve(BorderedLinearProblem(J, b, c, d, f, g), BorderingBLS(; k)), (0, 3))
        end

        err(s) = norm(s.v - vref) / norm(vref) + abs(s.σ - σref) / abs(σref)
        println("  nearly singular: k=0 err=", err(plain), "  k=3 err=", err(corrected))
        @test err(corrected) <= err(plain)
    end

    @testset "a wide border" begin
        # A wide border is a tuple of states. It is not inferred from an array's shape,
        # because a state can itself be a matrix. Bordering makes m inner solves for the
        # border block, then a small m x m solve; DirectBLS assembles the whole thing.
        n, m = 8, 2
        for (inner, rtol) in INNERS, alg in bordered_algorithms(inner)
            J = rand(n, n) + n * I
            bt = ntuple(_ -> rand(n), m)
            ct = ntuple(_ -> rand(n), m)
            d = rand(m, m)
            f = rand(n); g = rand(m)
            assembled = [J reduce(hcat, collect(bt)); reduce(hcat, collect(ct))' d]
            cache = init(BorderedLinearProblem(J, bt, ct, d, f, g), alg)
            sol = leaving_intact(() -> solve!(cache), J, bt..., ct..., d, f, g)
            xref = assembled \ vcat(f, g)
            @test successful_retcode(sol)
            @test sol.v ≈ xref[1:n] rtol = rtol
            @test sol.σ ≈ xref[(n + 1):end] rtol = rtol

            f2, g2 = rand(n), rand(m)
            reinit!(cache; f = f2, g = g2)
            @test !cache.inner.isfresh
            sol = leaving_intact(() -> solve!(cache), J, bt..., ct..., d, f2, g2)
            xref = assembled \ vcat(f2, g2)
            @test sol.v ≈ xref[1:n] rtol = rtol
            @test sol.σ ≈ xref[(n + 1):end] rtol = rtol
        end
    end

    @testset "a border may be one state, a tuple of states or a vector of states" begin
        # At width one the correction sweeps took `c` itself for the state, so a tuple or a
        # vector holding one state threw a `DimensionMismatch` once `k` was above zero.
        n = 6
        for (inner, rtol) in INNERS, alg in bordered_algorithms(inner), m in (1, 2), form in (:state, :tuple, :vector)
            form === :state && m > 1 && continue
            J = rand(n, n) + n * I; f = rand(n)
            bs = [rand(n) for _ in 1:m]; cs = [rand(n) for _ in 1:m]
            border(x) = form === :state ? only(x) : form === :tuple ? Tuple(x) : x
            d = m == 1 ? 1.5 : rand(m, m) + m * I; g = m == 1 ? 0.7 : rand(m)
            sol = leaving_intact(J, bs..., cs..., f) do
                solve(BorderedLinearProblem(J, border(bs), border(cs), d, f, g), alg)
            end
            x = [J reduce(hcat, bs); reduce(hcat, cs)' d] \ vcat(f, g)
            @test successful_retcode(sol)
            @test sol.v ≈ x[1:n] rtol = rtol
            @test sol.σ ≈ (m == 1 ? x[end] : x[(n + 1):end]) rtol = rtol
        end
    end

    @testset "a failed update leaves the cache solving the old problem" begin
        # The new problem was stored before the refresh ran. When the refresh threw, the next
        # solve! paired it with the old factorization: a wrong answer with a Success retcode.
        # After its first refactor DirectBLS factors its matrix in place, so a refresh that
        # fails partway has also overwritten part of the old factors.
        n = 6
        for (inner, rtol) in INNERS, alg in bordered_algorithms(inner), update in (:reinit, :prob)
            J = rand(n, n) + n * I
            b = rand(n); c = rand(n); d = 1.5; f = rand(n); g = 0.7
            # Bordering hands `J` to the inner solver as it is; only a factorization reads it in
            # a refresh, to copy it. The default picks a Krylov solver for this matrix type.
            alg isa BorderingBLS && !(inner isa AbstractFactorization) && continue
            cache = init(BorderedLinearProblem(BreakableMatrix(J, false), b, c, d, f, g), alg)
            solve!(cache)
            reinit!(cache; J = BreakableMatrix(J, false))
            solve!(cache)
            broken, f2 = BreakableMatrix(J + I, true), rand(n)
            fail!() = update === :reinit ? reinit!(cache; J = broken, f = f2) :
                (cache.prob = BorderedLinearProblem(broken, b, c, d, f2, g))
            @test_throws ErrorException fail!()
            @test !cache.prob.J.broken
            test_reference(solve!(cache), J, b, c, d, f, g; rtol)
        end
    end

    @testset "a failed update rethrows the exception its refresh raised" begin
        # The update restores the old problem and rethrows. The caller must get the very object
        # the refresh threw, not one from reading the old `J` or a wrapper around it.
        n = 6
        for (inner, rtol) in INNERS, alg in bordered_algorithms(inner), update in (:reinit, :prob)
            # As above.
            alg isa BorderingBLS && !(inner isa AbstractFactorization) && continue
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
        for (inner, rtol) in INNERS, alg in bordered_algorithms(inner), update in (:reinit, :prob)
            # As above.
            alg isa BorderingBLS && !(inner isa AbstractFactorization) && continue
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
            test_reference(solve!(cache), J.data, b, c, d, f, g; rtol)
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
        for (inner, rtol) in INNERS, alg in bordered_algorithms(inner), rejected in (:type, :size, :prob)
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
            test_reference(solve!(cache), J, b, c, d, f, g; rtol)
        end
    end

    @testset "a failed update whose old J cannot be read leaves no stale factors" begin
        # After its first refactor DirectBLS factors its assembled matrix in place, and an
        # assembly that throws partway leaves half-written factors. If the old `J` threw too,
        # they stayed marked current: a wrong answer with a Success retcode.
        n = 6
        aliasing = (NamedTuple(), (; alias = LinearAliasSpecifier(alias_A = true)))
        for (inner, rtol) in INNERS, alg in bordered_algorithms(inner), kwargs in aliasing,
                update in (:reinit, :prob), next in (:solve, :rhs)
            # As above, and aliasing would hand an in-place factorization the caller's read-only `J`.
            alg isa BorderingBLS && !(inner isa AbstractFactorization && isempty(kwargs)) && continue
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
            test_reference(solve!(cache), J, b, c, d, f, g; rtol)
            # Refactored, the cache keeps its factorization for a new right-hand side again.
            reinit!(cache; f = rand(n))
            @test !cache.inner.isfresh
        end
    end

    @testset "an update may change the state size or the border width" begin
        # The caches kept their buffers for the old sizes. Bordering then threw at the next
        # solve!, and DirectBLS rewrote only part of its old matrix and returned a wrong answer
        # with a Success retcode.
        for (inner, rtol) in INNERS, alg in bordered_algorithms(inner), update in (:reinit, :prob)
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
                sol = leaving_intact(() -> solve!(cache), J2, b, c, f)
                test_reference(sol, J2, b, c, d, f, g; rtol)
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
                sol = leaving_intact(() -> solve!(cache), J, b..., c..., d, f, g)
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
        for (inner, rtol) in INNERS, alg in bordered_algorithms(inner), update in (:reinit, :prob)
            # Only a Krylov solver takes the matrix-free `J`. Assembly needs a matrix instead, and
            # applies it to the flattened state, whatever its shape.
            alg isa BorderingBLS && inner isa AbstractFactorization && continue
            J = alg isa BorderingBLS ? RowScaling() : rand(6, 6) + 6I
            # `J` on the flattened 3 x 2 state.
            Jflat = alg isa BorderingBLS ? kron(I(2), Diagonal(1.0:3.0)) : J
            cache = init(BorderedLinearProblem(J, rand(2, 3), rand(2, 3), 1.5, rand(2, 3), 0.7), alg)
            solve!(cache)
            b = rand(3, 2); c = rand(3, 2); f = rand(3, 2)
            update === :reinit ? reinit!(cache; b, c, f) :
                (cache.prob = BorderedLinearProblem(J, b, c, 1.5, f, 0.7))
            sol = leaving_intact(() -> solve!(cache), b, c, f)
            @test size(sol.v) == (3, 2)
            test_reference(sol, Jflat, b, c, 1.5, f, 0.7; rtol)
        end
    end

    @testset "an update cannot change the type of a part" begin
        # A cache's fields are typed at `init`. A tuple border of another length, or a wide
        # border for a single one, threw a `TypeError` from `setfield!` or a `MethodError`.
        n = 5
        for (inner, _) in INNERS, alg in bordered_algorithms(inner), (m1, m2) in ((2, 3), (1, 2), (2, 1)),
                update in (:reinit, :prob)
            old, new = random_parts(n, m1), random_parts(n, m2)
            cache = init(BorderedLinearProblem(old...), alg)
            sol = solve!(cache)
            # A rejected update that would have refactored leaves the cache stale; see the testset above.
            names = filter(!=(:stale), fieldnames(typeof(cache)))
            fields = map(name -> getfield(cache, name), names)
            @test_throws ArgumentError update === :reinit ? reinit!(cache; new.b, new.c, new.d, new.g) :
                (cache.prob = BorderedLinearProblem(old.J, new.b, new.c, new.d, old.f, new.g))
            # Nothing else changed: the cache holds the same objects and solves the old problem as before.
            @test all(map(===, fields, map(name -> getfield(cache, name), names)))
            @test !cache.inner.isfresh
            sol2 = solve!(cache)
            @test sol2.v == sol.v
            @test sol2.σ == sol.σ
        end
        # The error names the parts whose type changed and points to `init`.
        cache = init(BorderedLinearProblem(random_parts(n, 1)...), BorderingBLS())
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
        sol2 = leaving_intact(() -> solve(prob2, BorderingBLS(KrylovJL_GMRES())), b2, c2, f2)

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

    @testset "det_sign is the sign of the assembled matrix's determinant" begin
        # Random matrices without a dominant diagonal, so both signs come up.
        n = 6
        for element in (Float32, Float64, ComplexF64), m in (1, 2), alg in bordered_algorithms(LUFactorization(); det_sign = true)
            signs = map(1:16) do seed
                rng = Xoshiro(seed)
                state() = randn(rng, element, n)
                border() = m == 1 ? state() : ntuple(_ -> state(), m)
                J, b, c = randn(rng, element, n, n), border(), border()
                d = m == 1 ? randn(rng, element) : randn(rng, element, m, m)
                g = m == 1 ? randn(rng, element) : randn(rng, element, m)
                cache = init(BorderedLinearProblem(J, b, c, d, state(), g), alg)
                @test successful_retcode(solve!(cache))
                @test det_sign(cache) ≈ sign(det(assembled(J, b, c, d))) rtol = sqrt(eps(real(element)))
                @test det_sign(cache) isa element
                det_sign(cache)
            end
            element <: Real && @test sort(unique(signs)) == [-1, 1]
        end
        # A complex border on a real J makes the Schur complement complex. Correction sweeps
        # would hand the real inner cache a complex right-hand side, so they stay out.
        rng = Xoshiro(0)
        J, b, c = randn(rng, n, n), randn(rng, n), randn(rng, ComplexF64, n)
        for alg in (BorderingBLS(; inner = LUFactorization(), det_sign = true), DirectBLS(; inner = LUFactorization(), det_sign = true))
            cache = init(BorderedLinearProblem(J, b, c, 0.5, randn(rng, n), 1.0 + 0im), alg)
            @test successful_retcode(solve!(cache))
            @test det_sign(cache) ≈ sign(det(assembled(J, b, c, 0.5)))
            @test det_sign(cache) isa ComplexF64
        end
        # A sparse J's LU, UMFPACK's under bordering, has a determinant too.
        n = 60
        J = spdiagm(-1 => -ones(n - 1), 0 => 4ones(n), 1 => -ones(n - 1))
        b, c, d = rand(n), rand(n), -1.0e3
        for alg in bordered_algorithms(LUFactorization(); det_sign = true)
            cache = init(BorderedLinearProblem(J, b, c, d, rand(n), 0.25), alg)
            solve!(cache)
            @test det_sign(cache) == sign(det(assembled(J, b, c, d))) == -1
        end
    end

    @testset "det_sign keeps its sign through a fold of J" begin
        # J = diag(t, 1, ...) is singular at t = 0, where det J = t and the Schur complement
        # -1/t both change sign. The bordered determinant stays -1 on both sides.
        n = 4
        for element in (Float32, Float64), alg in bordered_algorithms(LUFactorization(); det_sign = true), t in (1.0e-3, -1.0e-3)
            e1 = [one(element); zeros(element, n - 1)]
            J = Matrix(Diagonal([element(t); ones(element, n - 1)]))
            cache = init(BorderedLinearProblem(J, e1, e1, zero(element), ones(element, n), one(element)), alg)
            @test successful_retcode(solve!(cache))
            @test det_sign(cache) == -1
        end
    end

    @testset "det_sign is zero for a singular Schur complement whose solve succeeds" begin
        # With J = I and b = c = e1, d = 1 makes d - c'J⁻¹b = 0. Bordering divides by it and
        # reports success, while DirectBLS fails to factor the singular matrix.
        n = 4
        e1 = [1.0; zeros(n - 1)]
        for alg in (BorderingBLS(; inner = LUFactorization(), det_sign = true), BorderingBLS(; inner = LUFactorization(), k = 2, det_sign = true))
            cache = init(BorderedLinearProblem(Matrix(1.0I, n, n), e1, e1, 1.0, rand(n), 0.5), alg)
            @test successful_retcode(solve!(cache))
            @test det_sign(cache) == 0
        end
    end

    @testset "det_sign needs a successful solve! since the last update" begin
        for alg in bordered_algorithms(LUFactorization(); det_sign = true), m in (1, 2)
            parts = random_parts(5, m)
            cache = init(BorderedLinearProblem(parts...), alg)
            @test_throws "successful solve!" det_sign(cache)
            solve!(cache)
            @test det_sign(cache) == sign(det(assembled(parts.J, parts.b, parts.c, parts.d)))
            # A rejected update changes nothing, so the sign of the last solve stays.
            @test_throws ArgumentError reinit!(cache; d = m == 1 ? 2 : [2 0; 0 2])
            @test det_sign(cache) == sign(det(assembled(parts.J, parts.b, parts.c, parts.d)))
            reinit!(cache; f = rand(5))
            @test_throws "successful solve!" det_sign(cache)
            solve!(cache)
            cache.prob = BorderedLinearProblem(parts...)
            @test_throws "successful solve!" det_sign(cache)
            # A singular J fails its LU, and a solve that fails forgets the sign of the last one.
            solve!(cache)
            reinit!(cache; J = zeros(5, 5))
            @test !successful_retcode(solve!(cache))
            @test_throws "successful solve!" det_sign(cache)
        end
    end

    @testset "a solve! that throws leaves no sign" begin
        # Bordering's correction sweeps read J, and DirectBLS reads the state, after the factors
        # are in place. A broken one throws there, partway through the solve.
        n = 4
        J = BreakableMatrix(rand(n, n) + n * I, false)
        sweeping = BorderedLinearProblem(J, rand(n), rand(n), 1.5, rand(n), 0.7)
        f = BreakableMatrix(rand(2, 2), false)
        assembling = BorderedLinearProblem(rand(n, n) + n * I, rand(2, 2), rand(2, 2), 1.5, f, 0.7)
        for (prob, alg, part) in (
                (sweeping, BorderingBLS(; inner = LUFactorization(), k = 2, det_sign = true), J),
                (assembling, DirectBLS(; inner = LUFactorization(), det_sign = true), f),
            )
            part.broken = false
            cache = init(prob, alg)
            @test successful_retcode(solve!(cache))
            det_sign(cache)
            part.broken = true
            @test_throws part.err solve!(cache)
            @test_throws "successful solve!" det_sign(cache)
        end
    end

    @testset "det_sign is read at the solve, before aliased factors can change" begin
        # With `alias_A`, LU factors J in place, so the factors live in the caller's array.
        n = 5
        for alg in bordered_algorithms(LUFactorization(); det_sign = true)
            parts = random_parts(n, 1)
            cache = init(BorderedLinearProblem(parts...), alg; alias = LinearAliasSpecifier(alias_A = true))
            expected = sign(det(assembled(parts.J, parts.b, parts.c, parts.d)))
            @test successful_retcode(solve!(cache))
            # Negating every factor flips the sign of the product of the 5 pivots of U.
            parts.J .*= -1
            @test det_sign(cache) == expected
        end
    end

    @testset "det_sign = true needs inner = LUFactorization(), and det_sign needs det_sign = true" begin
        # A normal-equation solver keeps factors of A'A, whose determinant is never negative.
        prob = BorderedLinearProblem(random_parts(5, 1)...)
        for inner in (first.(INNERS)..., NormalCholeskyFactorization())
            for alg in bordered_algorithms(inner; det_sign = true)
                inner isa LUFactorization || @test_throws "det_sign = true needs inner = LUFactorization()" init(prob, alg)
            end
            # Without the option, every algorithm solves as before and keeps no sign.
            for alg in bordered_algorithms(inner)
                cache = init(prob, alg)
                @test successful_retcode(solve!(cache))
                @test_throws "built with det_sign = true" det_sign(cache)
            end
        end
    end
end
