# SupernodalLU allocation coverage.
#
# Separate from `qa/allocations.jl` because the two are included by different
# groups: `allocations.jl` also runs under `[AppleAccelerate]`, which covers
# `pre`, while this file is QA-only.  The direct-BLAS refactorization proofs
# over there hold on every release; the sweep proof here does not, because
# above `PANEL_BLAS_CUTOFF` the sweeps hand off to LinearAlgebra's
# `ldiv!`/`mul!`, whose wrappers are not statically clean on every version.
# Keeping them in one file put a 1.12-only proof inside a file that `pre` runs.

using AllocCheck, LinearAlgebra, LinearSolve, SparseArrays, Test

# The triangular sweeps run entirely off buffers owned by the factorization,
# whose sizes are known from the symbolic analysis, so they carry no growth
# branch and AllocCheck can prove them allocation-free statically (rather than
# sampling `@allocated`).  The user-facing `solve!` keeps a one-time scratch
# sizing, so it is asserted at runtime instead.
@check_allocs allocation_checked_supernodal_sweeps!(y, F) =
    LinearSolve.SupernodalLU._solve_panels!(y, F)

# The *static proof* is version-dependent: 1.10 and 1.13 leave a
# `generic_trimatdiv!` dynamic dispatch plus boxed `MulAddMul`/`SubArray` values
# that AllocCheck reports (`triangular.jl`, `matmul.jl` — stdlib frames, no
# SupernodalLU code involved); 1.12 folds all of it away.  Measured identically
# on x86_64-Linux and aarch64-Darwin (18 findings on 1.10 on both, 0 on 1.12),
# so this is the stdlib, not the platform, and not this package's code.
#
# The *runtime* assertions below are deliberately NOT gated: those measure real
# allocations, and skipping one would hide a genuine regression rather than a
# limitation of the prover.
const STATIC_SWEEP_PROOF = v"1.12" <= VERSION < v"1.13"

function poisson2d_qa(k)
    n = k * k
    Is = Int[]; Js = Int[]; V = Float64[]
    idx(i, j) = (j - 1) * k + i
    for j in 1:k, i in 1:k
        c = idx(i, j)
        push!(Is, c); push!(Js, c); push!(V, 4.0)
        i > 1 && (push!(Is, c); push!(Js, idx(i - 1, j)); push!(V, -1.0))
        i < k && (push!(Is, c); push!(Js, idx(i + 1, j)); push!(V, -1.0))
        j > 1 && (push!(Is, c); push!(Js, idx(i, j - 1)); push!(V, -1.0))
        j < k && (push!(Is, c); push!(Js, idx(i, j + 1)); push!(V, -1.0))
    end
    return sparse(Is, Js, V, n, n)
end

@testset "SupernodalLU solve sweeps are provably allocation-free" begin
    for A in (
            SparseArrays.spdiagm(
                0 => fill(4.0, 200), 1 => fill(-1.0, 199), -1 => fill(-1.0, 199)
            ),
            poisson2d_qa(30),                 # real panels: maxnu > 1
        )
        n = size(A, 1)
        F = LinearSolve.SupernodalLU.snlu(A)
        LinearSolve.SupernodalLU._ensure_panel_scratch!(F, 1)
        y = ones(n)
        if STATIC_SWEEP_PROOF
            allocation_checked_supernodal_sweeps!(y, F)   # throws if it can allocate
            @test true
        end
        # the full solve!, which owns the one-time sizing, is zero at runtime
        b = ones(n)
        x = similar(b)
        LinearSolve.SupernodalLU.solve!(x, F, b; refine = 0)
        @test @allocated(LinearSolve.SupernodalLU.solve!(x, F, b; refine = 0)) == 0
        @test norm(A * x - b) <= 1.0e-8 * norm(b)
        # multi-RHS reuses the factor-owned scratch once sized
        B = ones(n, 3)
        X = similar(B)
        LinearSolve.SupernodalLU.solve!(X, F, B; refine = 0)
        @test @allocated(LinearSolve.SupernodalLU.solve!(X, F, B; refine = 0)) == 0
    end
end
