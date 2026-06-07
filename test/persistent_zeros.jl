using LinearSolve, SparseArrays, LinearAlgebra, Test

# A family of sparse matrices sharing a FIXED stored sparsity pattern, where many
# stored entries are zero in every step (persistently dead) and a few "wobble"
# (zero in some steps, nonzero in others) -- the regime the
# `persistent_nonstructural_zeros` operator assumption targets.
function pz_pattern(n)
    I = Int[]
    J = Int[]
    for j in 1:n
        for i in max(1, j - 1):min(n, j + 1)
            push!(I, i)
            push!(J, j)
        end
        j <= n - 2 && (push!(I, j + 2); push!(J, j))
    end
    return sparse(I, J, ones(length(I)), n, n)
end

function pz_step(P, n, step; wobble_cols = (2, 5), singular = false)
    cp = SparseArrays.getcolptr(P)
    rv = rowvals(P)
    nz = zeros(length(rv))
    @inbounds for j in 1:n
        for p in cp[j]:(cp[j + 1] - 1)
            i = rv[p]
            if singular && j == 1
                nz[p] = 0.0   # zero the whole first column => structurally singular
            elseif i == j
                nz[p] = 4.0
            elseif abs(i - j) == 1
                nz[p] = -1.0
            elseif i == j + 2
                nz[p] = (j in wobble_cols) ? (isodd(step + j) ? 0.0 : 0.3) : 0.0
            end
        end
    end
    return SparseMatrixCSC(n, n, copy(cp), copy(rv), nz)
end

reduction(cache) = cache.cacheval.sparse_reduction

@testset "persistent_nonstructural_zeros (sparse reduction)" begin
    n = 12
    P = pz_pattern(n)
    nsteps = 8
    mats = [pz_step(P, n, s) for s in 1:nsteps]
    b = collect(1.0:n)
    refs = [Matrix(A) \ b for A in mats]

    @test count(iszero, nonzeros(mats[1])) > 0
    @test all(A -> SparseArrays.getcolptr(A) == SparseArrays.getcolptr(mats[1]), mats)

    @testset "force on: correctness + caching" begin
        cache = init(
            LinearProblem(copy(mats[1]), copy(b));
            assumptions = OperatorAssumptions(true; persistent_nonstructural_zeros = true)
        )
        for (i, A) in enumerate(mats)
            cache.A = copy(A)
            cache.b = copy(b)
            sol = solve!(cache)
            @test sol.retcode == ReturnCode.Success
            @test sol.u ≈ refs[i] rtol = 1.0e-8
        end
        red = reduction(cache)
        @test red.active
        @test length(red.keep) < length(red.mask)          # dropped persistent zeros
        @test red.nanalyze <= 4                              # bounded widenings, not per-step
        @test red.nrefactor >= nsteps - red.nanalyze
        a0 = red.nanalyze                                    # converged: no further widening
        for A in mats
            cache.A = copy(A)
            cache.b = copy(b)
            solve!(cache)
        end
        @test red.nanalyze == a0
    end

    @testset "force off: inactive, matches plain solve" begin
        cache = init(
            LinearProblem(copy(mats[1]), copy(b));
            assumptions = OperatorAssumptions(true; persistent_nonstructural_zeros = false)
        )
        for (i, A) in enumerate(mats)
            cache.A = copy(A)
            cache.b = copy(b)
            @test solve!(cache).u ≈ refs[i] rtol = 1.0e-8
        end
        @test !reduction(cache).active
    end

    @testset "auto-detect from the starting matrix" begin
        # many stored zeros => activates
        on = init(LinearProblem(copy(mats[1]), copy(b)))            # default assumptions = auto
        solve!(on)
        @test reduction(on).active
        # tight matrix (no stored zeros) => stays inactive (no overhead, bit-identical)
        tight = dropzeros(copy(mats[1]))
        off = init(LinearProblem(tight, copy(b)))
        solve!(off)
        @test !reduction(off).active
    end

    @testset "reduced operand reaches the singular QR fallback" begin
        sing = [pz_step(P, n, s; singular = true) for s in 1:3]
        cache = init(
            LinearProblem(copy(sing[1]), copy(b));
            assumptions = OperatorAssumptions(true; persistent_nonstructural_zeros = true)
        )
        for A in sing
            cache.A = copy(A)
            cache.b = copy(b)
            sol = solve!(cache)
            @test sol.retcode == ReturnCode.Success          # QR fallback, not an error
            @test cache.cacheval.fell_back_to_qr             # the LU→QR fallback fired
            @test reduction(cache).active
            # the reduced pattern is smaller than the full stored pattern, and that
            # is the operand the swap hands to the LU attempt AND the QR fallback
            @test length(reduction(cache).keep) < length(reduction(cache).mask)
        end
    end

    @testset "inconsistent zeros: auto switches to per-solve dropzeros" begin
        # Fixed stored pattern = diagonal + 5 super-diagonal bands, but each step
        # only ONE band is nonzero (rotating). The union of ever-nonzero positions
        # grows to the full pattern, so the cached-union reduction bloats; auto must
        # switch to dropping each matrix's own zeros per solve.
        m = 16
        bands = 1:5
        I = Int[]
        J = Int[]
        for j in 1:m
            push!(I, j); push!(J, j)
            for d in bands
                j + d <= m && (push!(I, j + d); push!(J, j))
            end
        end
        Pin = sparse(I, J, ones(length(I)), m, m)
        function inc_step(P, m, step)
            cp = SparseArrays.getcolptr(P)
            rv = rowvals(P)
            nz = zeros(length(rv))
            active_band = (step % 5) + 1
            @inbounds for j in 1:m
                for p in cp[j]:(cp[j + 1] - 1)
                    i = rv[p]
                    nz[p] = i == j ? 10.0 : (i - j == active_band ? -1.0 : 0.0)
                end
            end
            return SparseMatrixCSC(m, m, copy(cp), copy(rv), nz)
        end
        steps = [inc_step(Pin, m, s) for s in 0:11]
        bin = collect(1.0:m)
        cache = init(LinearProblem(copy(steps[1]), copy(bin)))   # auto
        for A in steps
            cache.A = copy(A)
            cache.b = copy(bin)
            sol = solve!(cache)
            @test sol.u ≈ Matrix(A) \ bin rtol = 1.0e-8
        end
        red = reduction(cache)
        @test red.active
        @test !red.cache_union          # bloated union => switched to per-solve dropzeros
    end

    @testset "standalone sparse solvers apply the reduction" begin
        algs = Any[PureKLUFactorization(), KLUFactorization()]
        Base.USE_GPL_LIBS && push!(algs, UMFPACKFactorization())
        @testset "$(nameof(typeof(alg)))" for alg in algs
            cache = init(
                LinearProblem(copy(mats[1]), copy(b)), alg;
                assumptions = OperatorAssumptions(true; persistent_nonstructural_zeros = true)
            )
            @test cache.sparse_reduction !== nothing
            for (i, A) in enumerate(mats)
                cache.A = copy(A)
                cache.b = copy(b)
                @test solve!(cache).u ≈ refs[i] rtol = 1.0e-8
            end
            red = cache.sparse_reduction
            @test red.active
            @test length(red.keep) < length(red.mask)   # reduction really happened
        end
        # force off => standalone reduction inactive, plain solve
        coff = init(
            LinearProblem(copy(mats[1]), copy(b)), PureKLUFactorization();
            assumptions = OperatorAssumptions(true; persistent_nonstructural_zeros = false)
        )
        solve!(coff)
        @test !coff.sparse_reduction.active
    end

    @testset "non-square least-squares (default sparse QR) drops zeros" begin
        m, k = 14, 9
        rows = Int[]
        cols = Int[]
        for j in 1:k, i in (j, j + 1, j + 5)
            i <= m && (push!(rows, i); push!(cols, j))
        end
        Pr = sparse(rows, cols, ones(length(rows)), m, k)
        function ls_step(P, s)
            cp = SparseArrays.getcolptr(P)
            rv = rowvals(P)
            nz = zeros(length(rv))
            for j in 1:size(P, 2), p in cp[j]:(cp[j + 1] - 1)
                i = rv[p]
                nz[p] = i == j ? 3.0 : (i == j + 1 ? 1.0 : (j == 2 && isodd(s) ? 0.5 : 0.0))
            end
            return SparseMatrixCSC(size(P, 1), size(P, 2), copy(cp), copy(rv), nz)
        end
        lsmats = [ls_step(Pr, s) for s in 1:6]
        bls = ones(m)
        cache = init(
            LinearProblem(copy(lsmats[1]), copy(bls));
            assumptions = OperatorAssumptions(false; persistent_nonstructural_zeros = true)
        )
        for A in lsmats
            cache.A = copy(A)
            cache.b = copy(bls)
            @test solve!(cache).u ≈ Matrix(A) \ bls rtol = 1.0e-8   # vs dense least squares
        end
        red = cache.cacheval.sparse_reduction
        @test red.active
        @test size(red.reduced) == (m, k)                # reduced keeps the rectangular shape
        @test length(red.keep) < length(red.mask)        # zeros really dropped
    end

    @testset "constant-pattern contract is enforced" begin
        cache = init(
            LinearProblem(copy(mats[1]), copy(b));
            assumptions = OperatorAssumptions(true; persistent_nonstructural_zeros = true)
        )
        solve!(cache)
        cache.A = sparse(2.0I, n, n)   # different stored pattern
        @test_throws ArgumentError solve!(cache)
    end
end
