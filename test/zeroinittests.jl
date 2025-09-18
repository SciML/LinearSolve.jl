using LinearSolve, LinearAlgebra, SparseArrays, Test

A = Diagonal(ones(4))
b = rand(4)
A = sparse(A)
Anz = copy(A)
C = copy(A)
C[begin, end] = 1e-8
A.nzval .= 0
cache_kwargs = (; verbose = true, abstol = 1e-8, reltol = 1e-8, maxiter = 30)

function test_nonzero_init(alg = nothing)
    linprob = LinearProblem(A, b)

    cache = init(linprob, alg)
    cache.A = Anz
    sol = solve!(cache; cache_kwargs...)
    @test sol.u == b
    cache.A = C
    sol = solve!(cache; cache_kwargs...)
    @test sol.u ≈ b
end

test_nonzero_init()
test_nonzero_init(KLUFactorization())
test_nonzero_init(UMFPACKFactorization())
