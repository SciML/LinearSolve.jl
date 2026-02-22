using LinearSolve, SparseArrays, LinearAlgebra, Test
using Ginkgo

@testset "GinkgoJL" begin
    # Build a Float32 SPD sparse matrix (Ginkgo.jl requires Float32 / Int32)
    n = 20
    A = let B = sprandn(Float32, n, n, 0.3)
        SparseMatrixCSC{Float32, Int32}(B' * B + Float32(n) * I)
    end
    b = rand(Float32, n)
    prob = LinearProblem(A, b)

    # GinkgoJL_CG convenience alias
    @test GinkgoJL_CG() isa GinkgoJL
    sol = solve(prob, GinkgoJL_CG(); reltol = 1.0f-4, maxiters = 500)
    @test norm(A * sol.u - b) / norm(b) < 5.0f-3

    # GinkgoJL_GMRES: constructor returns a GinkgoJL, but solve errors until
    # Ginkgo.jl exposes a GMRES solver
    @test GinkgoJL_GMRES() isa GinkgoJL
    @test_throws ErrorException solve(prob, GinkgoJL_GMRES(); reltol = 1.0f-4, maxiters = 500)
end
