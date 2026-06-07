using LinearSolve, SparseArrays, LinearAlgebra, Test

# A family of sparse matrices sharing a FIXED stored sparsity pattern, where many
# stored entries are zero in every step (persistently dead) and a few "wobble"
# (zero in some steps, nonzero in others). This is the regime PersistentDrop targets.
#
# Pattern (per column j): diagonal (always 4), first off-diagonals (always -1), and
# an extra slot at row (j+2). The extra slot is a persistent zero except for j in
# `wobble_cols`, where it wobbles between 0 and a small nonzero across steps.
function pd_pattern(n)
    I = Int[]
    J = Int[]
    for j in 1:n
        for i in max(1, j - 1):min(n, j + 1)
            push!(I, i)
            push!(J, j)
        end
        if j <= n - 2
            push!(I, j + 2)
            push!(J, j)
        end
    end
    return sparse(I, J, ones(length(I)), n, n)   # pattern carrier (all ones)
end

function pd_step(P, n, step; wobble_cols = (2, 5), singular = false)
    cp = SparseArrays.getcolptr(P)
    rv = rowvals(P)
    nz = zeros(length(rv))
    @inbounds for j in 1:n
        for p in cp[j]:(cp[j + 1] - 1)
            i = rv[p]
            if singular && j == 1
                nz[p] = 0.0   # zero the entire first column => structurally singular
            elseif i == j
                nz[p] = 4.0
            elseif abs(i - j) == 1
                nz[p] = -1.0
            elseif i == j + 2
                # extra slot: wobble for selected cols, persistent zero otherwise
                nz[p] = (j in wobble_cols) ? (isodd(step + j) ? 0.0 : 0.3) : 0.0
            end
        end
    end
    return SparseMatrixCSC(n, n, copy(cp), copy(rv), nz)
end

@testset "PersistentDropFactorization" begin
    n = 12
    P = pd_pattern(n)
    nsteps = 8
    mats = [pd_step(P, n, s) for s in 1:nsteps]
    b = collect(1.0:n)
    refs = [Matrix(A) \ b for A in mats]   # dense reference

    # there must be genuine stored zeros to drop, and a constant stored pattern
    @test all(A -> SparseArrays.getcolptr(A) == SparseArrays.getcolptr(mats[1]), mats)
    @test all(A -> rowvals(A) == rowvals(mats[1]), mats)
    @test count(iszero, nonzeros(mats[1])) > 0

    inner_algs = Any[
        ("default", PersistentDropFactorization()),
        ("PureKLU", PersistentDropFactorization(PureKLUFactorization())),
    ]
    if Base.USE_GPL_LIBS
        push!(inner_algs, ("UMFPACK", PersistentDropFactorization(UMFPACKFactorization())))
    end

    @testset "correctness + caching ($name)" for (name, alg) in inner_algs
        cache = init(LinearProblem(copy(mats[1]), copy(b)), alg)
        for (i, A) in enumerate(mats)
            cache.A = copy(A)
            cache.b = copy(b)
            sol = solve!(cache)
            @test sol.retcode == ReturnCode.Success
            @test sol.u ≈ refs[i] rtol = 1.0e-8
        end
        st = cache.cacheval
        # union mask should converge: dropped some persistent zeros, kept fewer than full
        @test length(st.keep) < length(st.mask)
        # widening (analyze) count is bounded by the number of wobble activations + 1,
        # and is far less than the number of steps (no per-step re-analysis)
        @test st.nanalyze <= 4
        @test st.nrefactor >= nsteps - st.nanalyze
        # re-feeding the converged sequence performs NO further widening
        a0 = st.nanalyze
        for A in mats
            cache.A = copy(A)
            cache.b = copy(b)
            solve!(cache)
        end
        @test st.nanalyze == a0
    end

    @testset "reduced operand reaches the singular QR fallback" begin
        sing = [pd_step(P, n, s; singular = true) for s in 1:3]
        cache = init(LinearProblem(copy(sing[1]), copy(b)), PersistentDropFactorization())
        local st
        for A in sing
            cache.A = copy(A)
            cache.b = copy(b)
            sol = solve!(cache)
            @test sol.retcode == ReturnCode.Success    # falls back to QR rather than erroring
            st = cache.cacheval
            inner = st.inner
            @test inner.cacheval isa LinearSolve.DefaultLinearSolverInit
            @test inner.cacheval.fell_back_to_qr        # the LU→QR fallback fired
            # the operand the inner (and thus its QR fallback) factorized is the REDUCED
            # matrix, not the full stored pattern
            @test nnz(inner.A) == length(st.keep)
            @test nnz(inner.A) < nnz(cache.A)
        end
    end

    @testset "constant-pattern contract is enforced" begin
        cache = init(LinearProblem(copy(mats[1]), copy(b)), PersistentDropFactorization())
        solve!(cache)
        # a matrix whose stored pattern differs is rejected rather than mis-solved
        bad = sparse(1.0I, n, n)   # different stored nnz / pattern
        cache.A = bad
        @test_throws ArgumentError solve!(cache)
    end
end
