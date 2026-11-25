using Enzyme: Enzyme, Const, Duplicated, Forward
using LinearAlgebra: Diagonal, UnitLowerTriangular, UnitUpperTriangular
using LinearSolve
using Reactant
using Test

# XLA hands julia_callbacks device buffers on CUDA-capable backends, which
# Reactant only wraps when its CUDA extension is loaded; this env has no CUDA.jl.
Reactant.set_default_backend("cpu")

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

@testset "rank-deficient least squares in the CPU callback" begin
    A_tall = Reactant.to_rarray(ones(Float32, 3, 2))
    b_tall = Reactant.to_rarray(Float32[1, 2, 3])
    @test Array(@jit solve_once(A_tall, b_tall)) ≈ Float32[1, 1]
end

@testset "native unit-triangular division" begin
    # The stored diagonal and unused triangle must not enter the solve.
    for wrap in (UnitLowerTriangular, UnitUpperTriangular), rhs in (b, Float64.(hcat(b, -b)))
        data = eltype(rhs)[0 2; 3 0]
        xla_solve(A, b) = solve(LinearProblem(wrap(A), b), DirectLdiv!())
        inputs = (Reactant.to_rarray(data), Reactant.to_rarray(rhs))
        sol = @jit xla_solve(inputs...)
        @test sol.alg isa DirectLdiv!
        @test sol.retcode == ReturnCode.Success
        @test Array(sol.u) ≈ wrap(data) \ rhs
        hlo = repr(@code_hlo optimize = false xla_solve(inputs...))
        @test occursin("stablehlo.triangular_solve", hlo)
        @test !occursin("reactant_julia_callback", hlo)
    end
end

function triangular_solution(A, b)
    return solve(LinearProblem(UnitLowerTriangular(A), b), DirectLdiv!()).u
end

@testset "forward derivative with respect to the right-hand side" begin
    derivative(A, b, db) = only(
        Enzyme.autodiff(Forward, triangular_solution, Const(A), Duplicated(b, db))
    )
    db = Float32[1, 0]
    @test Array(@jit derivative(A_reactant, b_reactant, Reactant.to_rarray(db))) ≈
        UnitLowerTriangular(A) \ db
end

@testset "unsupported unit-triangular request" begin
    other_alg(A, b) = solve(LinearProblem(UnitUpperTriangular(A), b), QRFactorization())
    @test_throws "QRFactorization" @jit other_alg(A_reactant, b_reactant)
    with_guess(A, b) = solve(LinearProblem(UnitLowerTriangular(A), b; u0 = b), DirectLdiv!())
    @test_throws "default solve options" @jit with_guess(A_reactant, b_reactant)
end

@testset "unsupported callback representation" begin
    wrapped(A, b) = solve(LinearProblem(Diagonal(A), b), LUFactorization())
    @test_throws "Julia callback requires dense traced buffers" @jit wrapped(b_reactant, b_reactant)
end
