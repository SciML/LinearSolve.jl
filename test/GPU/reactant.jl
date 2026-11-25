using CUDA, LinearSolve, Reactant, Test

Reactant.set_default_backend("gpu")
CUDA.allowscalar(false)

function solve_on_device(A, b, u, args...; kwargs...)
    if A isa CuMatrix && b isa CuVector && u isa CuVector
        return A \ b
    end
    return fill!(u, NaN32)
end

A = Float32[4 1; 2 3]
b = Float32[1, 2]
A_dev, b_dev = Reactant.to_rarray(A), Reactant.to_rarray(b)
for alg in (LUFactorization(), LinearSolveFunction(solve_on_device))
    sol = @jit solve(LinearProblem(A_dev, b_dev), alg)
    @test sol.alg === alg
    @test Array(sol.u) ≈ A \ b
end
