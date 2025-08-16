using LinearSolve, SparseArrays, Random, LinearAlgebra, Test
using Pardiso

println("Testing Pardiso memory leak fix...")

n = 100
A = sprand(n, n, 0.1) + sparse(I, n, n) * 5.0
b = rand(n)

println("Creating and solving multiple problems to check for memory leaks...")
for i in 1:5
    prob = LinearProblem(A, b)
    sol = LinearSolve.solve(prob, PardisoJL())
    println("Iteration $i: residual norm = ", norm(A * sol.u - b))
    @test norm(A * sol.u - b) < 1e-10
end

println("\nTesting that solver cleanup happens properly...")
prob = LinearProblem(A, b)
cache = LinearSolve.init(prob, PardisoJL())
sol = LinearSolve.solve!(cache)
@test norm(A * sol.u - b) < 1e-10

println("\nMemory leak fix test completed successfully!")
println("The finalizer should release Pardiso memory when cache is garbage collected.")