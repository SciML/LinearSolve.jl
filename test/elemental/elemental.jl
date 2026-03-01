using LinearSolve, LinearAlgebra, Test, Random, SciMLBase
using Elemental

@testset "ElementalJL" begin
    rng = Random.MersenneTwister(42)
    n = 50

    # ── General square Float64 system ─────────────────────────────────────────
    A_el = Matrix{Float64}(I, n, n) + 0.1 * rand(rng, n, n)
    b_el = rand(rng, n)
    prob_el = LinearProblem(A_el, b_el)

    # Default method: LU
    sol_lu = solve(prob_el, ElementalJL())
    @test A_el * sol_lu.u ≈ b_el atol=1e-8
    @test sol_lu.retcode == ReturnCode.Success

    # QR factorization
    sol_qr = solve(prob_el, ElementalJL(method = :QR))
    @test A_el * sol_qr.u ≈ b_el atol=1e-8

    # LQ factorization
    sol_lq = solve(prob_el, ElementalJL(method = :LQ))
    @test A_el * sol_lq.u ≈ b_el atol=1e-8

    # ── Symmetric positive definite system → Cholesky ─────────────────────────
    B_spd = rand(rng, n, n)
    A_spd = B_spd * B_spd' + n * I  # guaranteed SPD
    b_spd = rand(rng, n)
    sol_ch = solve(LinearProblem(A_spd, b_spd), ElementalJL(method = :Cholesky))
    @test A_spd * sol_ch.u ≈ b_spd atol=1e-8

    # ── Float32 system (all 4 eltypes are supported by Elemental_jll) ──────────
    A_f32 = Matrix{Float32}(I, n, n) + 0.1f0 * rand(rng, Float32, n, n)
    b_f32 = rand(rng, Float32, n)
    sol_f32 = solve(LinearProblem(A_f32, b_f32), ElementalJL())
    @test A_f32 * sol_f32.u ≈ b_f32 atol=1f-4

    # ── Complex system ─────────────────────────────────────────────────────────
    A_cplx = Matrix{ComplexF64}(I, n, n) + 0.1 * (rand(rng, n, n) + im * rand(rng, n, n))
    b_cplx = rand(rng, ComplexF64, n)
    sol_cplx = solve(LinearProblem(A_cplx, b_cplx), ElementalJL())
    @test A_cplx * sol_cplx.u ≈ b_cplx atol=1e-8

    # ── Pass an Elemental.Matrix directly as A ────────────────────────────────
    A_emat = convert(Elemental.Matrix{Float64}, A_el)
    b_emat = rand(rng, n)
    sol_emat = solve(LinearProblem(A_emat, b_emat), ElementalJL())
    @test A_el * sol_emat.u ≈ b_emat atol=1e-8

    # ── Elemental.Matrix as A: re-factorization must not use corrupted data ───
    # Regression: _to_elemental_matrix must copy A. If it returned A directly,
    # lu! would factorize in-place; the second reinit!(A=...) + solve! would
    # re-factorize already-destroyed data and produce a wrong result.
    A_emat2 = convert(Elemental.Matrix{Float64}, A_el)
    b_emat2a = rand(rng, n)
    b_emat2b = rand(rng, n)
    cache_emat = SciMLBase.init(LinearProblem(A_emat2, b_emat2a), ElementalJL())
    solve!(cache_emat)
    @test A_el * cache_emat.u ≈ b_emat2a atol=1e-8
    # Pass A= to reinit! so isfresh=true, forcing _to_elemental_matrix to run again.
    reinit!(cache_emat; A = A_emat2, b = b_emat2b)
    solve!(cache_emat)
    @test A_el * cache_emat.u ≈ b_emat2b atol=1e-8

    # ── Cache reuse (same A, different b) ─────────────────────────────────────
    b_el2 = rand(rng, n)
    cache = SciMLBase.init(prob_el, ElementalJL())
    solve!(cache)
    @test A_el * cache.u ≈ b_el atol=1e-8

    reinit!(cache; b = b_el2)
    solve!(cache)
    @test A_el * cache.u ≈ b_el2 atol=1e-8

    # ── Unknown method raises an error ────────────────────────────────────────
    @test_throws ErrorException solve(prob_el, ElementalJL(method = :Bogus))
end
