using LinearSolve
using Reactant
using Test

function solve_once(A, b)
    return solve(LinearProblem(A, b)).u
end

A = Float32[4 1; 2 3]
b = Float32[1, 2]
expected = A \ b
A_reactant = Reactant.to_rarray(A)
b_reactant = Reactant.to_rarray(b)

sol = @jit solve(LinearProblem(A_reactant, b_reactant))
@test Array(sol.u) ≈ expected
@test Array(@jit solve_once(A_reactant, b_reactant)) ≈ expected
@testset "preserves the LinearSolve operation" begin
    @test occursin(
        "reactant_julia_callback", repr(@code_hlo solve_once(A_reactant, b_reactant))
    )

    qr_sol = @jit solve(LinearProblem(A_reactant, b_reactant), QRFactorization())
    @test qr_sol.alg isa QRFactorization
    @test Array(qr_sol.u) ≈ expected
end

@testset "default LU preserves QR safety fallback" begin
    A_singular = Float32[1 1; 1 1]
    b_singular = Float32[1, 2]
    expected_singular = Float32[0.75, 0.75]

    lu_sol = solve(LinearProblem(A_singular, b_singular), LUFactorization())
    @test lu_sol.retcode == ReturnCode.Failure
    @test solve(LinearProblem(A_singular, b_singular)).u ≈ expected_singular

    A_singular_reactant = Reactant.to_rarray(A_singular)
    b_singular_reactant = Reactant.to_rarray(b_singular)
    lu_sol_reactant = @jit solve(
        LinearProblem(A_singular_reactant, b_singular_reactant), LUFactorization()
    )
    @test Array(lu_sol_reactant.u) == lu_sol.u

    sol_singular = @jit solve(
        LinearProblem(A_singular_reactant, b_singular_reactant)
    )
    @test Array(sol_singular.u) ≈ expected_singular
end
