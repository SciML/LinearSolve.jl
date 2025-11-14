module TestMKLLUFactorization
using LinearSolve
using LinearAlgebra
using StaticArrays
import SciMLBase

# Define a simple linear problem with a dense matrix
const A_matrix = [4.0 1.0; 1.0 3.0]
const b_vector = [1.0, 2.0]

const alg = MKLLUFactorization()
const prob = LinearProblem(A_matrix, b_vector)
const cache = init(prob, alg)

function solve_linear(x)
    # Create a new problem with a modified b vector
    b_new = [x, 2.0 * x]
    reinit!(cache, b_new)
    solve!(cache)
    return cache
end
end
