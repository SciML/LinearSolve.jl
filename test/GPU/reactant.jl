using CUDA, LinearSolve, Reactant, Test

Reactant.set_default_backend("gpu")
CUDA.allowscalar(false)

function solve_on_device(A, b, u, args...; kwargs...)
    if A isa CuMatrix && b isa CuVector && u isa CuVector
        return b .+ 10.0f0
    end
    return fill!(u, NaN32)
end

A = Float32[4 1; 2 3]
b = Float32[1, 2]
A_dev, b_dev = Reactant.to_rarray(A), Reactant.to_rarray(b)
for (alg, expected) in (
        (LUFactorization(), A \ b),
        (LinearSolveFunction(solve_on_device), Float32[11, 12]),
    )
    sol = @jit solve(LinearProblem(A_dev, b_dev), alg)
    @test sol.alg === alg
    @test Array(sol.u) ≈ expected
end
