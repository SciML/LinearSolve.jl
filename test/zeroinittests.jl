using LinearSolve, LinearAlgebra, SparseArrays, Test

A = Diagonal(ones(4))
b = rand(4)
A = sparse(A)
Anz = deepcopy(A)
A.nzval .= 0
cache_kwargs = (; verbose = true, abstol = 1e-8, reltol = 1e-8, maxiter = 30)


function test_nonzero_init(alg=nothing)
    linprob = LinearProblem(A,b)

    cache = init(linprob)
    cache = LinearSolve.set_A(cache, Anz)
    @show cache.A
    sol = solve(cache,nothing;cache_kwargs...)
    @test sol.u == b
end

test_nonzero_init()
test_nonzero_init(KLUFactorization())
test_nonzero_init(UMFPACKFactorization())
