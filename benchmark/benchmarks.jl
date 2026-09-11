using LinearSolve, BenchmarkTools
using LinearAlgebra, SparseArrays, StableRNGs

const SUITE = BenchmarkGroup()
const rng = StableRNG(123)

# =============================================================================
# Dense linear solves
# =============================================================================

SUITE["dense"] = BenchmarkGroup()

A = rand(rng, 300, 300)
A += 300I  # diagonal dominance for Krylov convergence
b = rand(rng, 300)
prob = LinearProblem(A, b)

SUITE["dense"]["default"] = @benchmarkable solve($prob)
SUITE["dense"]["LUFactorization"] = @benchmarkable solve($prob, LUFactorization())
SUITE["dense"]["QRFactorization"] = @benchmarkable solve($prob, QRFactorization())
SUITE["dense"]["KrylovJL_GMRES"] = @benchmarkable solve(
    $prob, KrylovJL_GMRES()
)
SUITE["dense"]["SimpleGMRES"] = @benchmarkable solve(
    $prob, SimpleGMRES(; blocksize = 30)
)

# =============================================================================
# Factorization reuse (init + solve! with changing RHS)
# =============================================================================

SUITE["reuse"] = BenchmarkGroup()

cache = init(prob, LUFactorization())
b2 = rand(rng, 300)

SUITE["reuse"]["init"] = @benchmarkable init($prob, LUFactorization())
SUITE["reuse"]["solve!"] = @benchmarkable solve!($cache)
SUITE["reuse"]["resolve_new_b"] = @benchmarkable begin
    c = deepcopy($cache)
    c.b = $b2
    solve!(c)
end

# =============================================================================
# Sparse linear solves
# =============================================================================

SUITE["sparse"] = BenchmarkGroup()

As = sprand(rng, 2000, 2000, 0.005)
As = As + As' + 2000I  # symmetric positive definite-ish
bs = rand(rng, 2000)
prob_sparse = LinearProblem(As, bs)

SUITE["sparse"]["default"] = @benchmarkable solve($prob_sparse)
SUITE["sparse"]["LUFactorization"] = @benchmarkable solve($prob_sparse, LUFactorization())
SUITE["sparse"]["KrylovJL_CG"] = @benchmarkable solve($prob_sparse, KrylovJL_CG())
SUITE["sparse"]["KrylovJL_GMRES"] = @benchmarkable solve(
    $prob_sparse, KrylovJL_GMRES()
)
