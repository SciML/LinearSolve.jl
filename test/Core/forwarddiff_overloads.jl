using LinearSolve
using ForwardDiff
using Test
using LinearAlgebra
using SparseArrays
using ComponentArrays
using Sparspak
using SpecializingFactorizations
using RecursiveFactorization

function h(p)
    return (
        A = [
            p[1] p[2] + 1 p[2]^3;
            3 * p[1] p[1] + 5 p[2] * p[1] - 4;
            p[2]^2 9 * p[1] p[2]
        ],
        b = [p[1] + 1, p[2] * 2, p[1]^2],
    )
end

# Opt-out dense methods: SpecializedLU/QR, GenericLU and RFLU solve the Dual
# problem directly (no primal/partials splitting). The partials can differ from
# `\` by an ulp, and isapprox over Dual vectors NaNs when the primal diff is
# exactly zero (sqrt has infinite slope at 0), so compare values and partials
# separately.
function dual_isapprox(x, y; rtol)
    isapprox(ForwardDiff.value.(x), ForwardDiff.value.(y); rtol) || return false
    return isapprox(
        reduce(hcat, collect.(ForwardDiff.partials.(x))),
        reduce(hcat, collect.(ForwardDiff.partials.(y))); rtol
    )
end

A, b = h([ForwardDiff.Dual(5.0, 1.0, 0.0), ForwardDiff.Dual(5.0, 0.0, 1.0)])
prob = LinearProblem(A, b)
backslash_x_p = A \ b
@test dual_isapprox(
    solve(prob, SpecializedLUFactorization()).u, backslash_x_p, rtol = 1.0e-9
)
@test dual_isapprox(
    solve(prob, SpecializedQRFactorization()).u, backslash_x_p, rtol = 1.0e-9
)
@test dual_isapprox(
    solve(prob, GenericLUFactorization()).u, backslash_x_p, rtol = 1.0e-9
)
@test dual_isapprox(
    solve(prob, RFLUFactorization()).u, backslash_x_p, rtol = 1.0e-9
)

# Rectangular least-squares with Duals through the direct dual path
p_ls = [ForwardDiff.Dual(2.0, 1.0, 0.0), ForwardDiff.Dual(3.0, 0.0, 1.0)]
A_ls = [p_ls[1] 1.0; 1.0 p_ls[2]; p_ls[1] p_ls[2]]
b_ls = [p_ls[1] + 1, p_ls[2] * 2, p_ls[1] * p_ls[2]]
qr_ls_x_p = solve(LinearProblem(A_ls, b_ls), SpecializedQRFactorization())
@test dual_isapprox(qr_ls_x_p.u, qr(A_ls) \ b_ls, rtol = 1.0e-9)

# Opt-out path with duals only in A (plain b) and only in b (plain A)

plain_b = ForwardDiff.value.(b)
prob = LinearProblem(A, plain_b)
@test ≈(solve(prob, GenericLUFactorization()), A \ plain_b, rtol = 1.0e-9)
@test ≈(solve(prob, RFLUFactorization()), A \ plain_b, rtol = 1.0e-9)

plain_A = ForwardDiff.value.(A)
prob = LinearProblem(plain_A, b)
@test ≈(solve(prob, GenericLUFactorization()), plain_A \ b, rtol = 1.0e-9)
@test ≈(solve(prob, RFLUFactorization()), plain_A \ b, rtol = 1.0e-9)

# Regression test for #1052: RFLUFactorization must stay on the split
# primal/partials path and NOT take the direct dual solve. Its fast Float64
# factorization is BLAS/SIMD-grade; routing the Dual problem through it falls
# back to generic scalar dual arithmetic (~40x slower). Guard the routing
# decision directly so RFLU is never re-added to the direct path.
@testset "RFLU stays off the direct dual path (#1052)" begin
    ext = Base.get_extension(LinearSolve, :LinearSolveForwardDiffExt)
    @test !ext._use_direct_dual_solve(RFLUFactorization())
    # Sanity: the genuinely-cheap-in-dual algorithms are still on the direct path.
    @test ext._use_direct_dual_solve(GenericLUFactorization())
    @test ext._use_direct_dual_solve(SpecializedLUFactorization())
    @test ext._use_direct_dual_solve(PureKLUFactorization())
end

# Overload Dense

A, b = h([ForwardDiff.Dual(5.0, 1.0, 0.0), ForwardDiff.Dual(5.0, 0.0, 1.0)])

prob = LinearProblem(A, b)
overload_x_p = solve(prob, LUFactorization())
backslash_x_p = A \ b
krylov_overload_x_p = solve(prob, KrylovJL_GMRES())
@test ≈(overload_x_p, backslash_x_p, rtol = 1.0e-9)
@test ≈(krylov_overload_x_p, backslash_x_p, rtol = 1.0e-9)

krylov_prob = LinearProblem(A, b, u0 = rand(3))
krylov_u0_sol = solve(krylov_prob, KrylovJL_GMRES())

@test ≈(krylov_u0_sol, backslash_x_p, rtol = 1.0e-9)

A, _ = h([ForwardDiff.Dual(5.0, 1.0, 0.0), ForwardDiff.Dual(5.0, 0.0, 1.0)])
backslash_x_p = A \ [6.0, 10.0, 25.0]
prob = LinearProblem(A, [6.0, 10.0, 25.0])

@test ≈(solve(prob).u, backslash_x_p, rtol = 1.0e-9)
@test ≈(solve(prob, KrylovJL_GMRES()).u, backslash_x_p, rtol = 1.0e-9)

_, b = h([ForwardDiff.Dual(5.0, 1.0, 0.0), ForwardDiff.Dual(5.0, 0.0, 1.0)])
A = [5.0 6.0 125.0; 15.0 10.0 21.0; 25.0 45.0 5.0]
backslash_x_p = A \ b
prob = LinearProblem(A, b)

@test ≈(solve(prob).u, backslash_x_p, rtol = 1.0e-9)
@test ≈(solve(prob, KrylovJL_GMRES()).u, backslash_x_p, rtol = 1.0e-9)

A, b = h([ForwardDiff.Dual(10.0, 1.0, 0.0), ForwardDiff.Dual(10.0, 0.0, 1.0)])

prob = LinearProblem(A, b)
cache = init(prob, LUFactorization())

new_A, new_b = h([ForwardDiff.Dual(5.0, 1.0, 0.0), ForwardDiff.Dual(5.0, 0.0, 1.0)])
cache.A = new_A
cache.b = new_b

@test cache.A == new_A
@test cache.b == new_b

x_p = solve!(cache)
backslash_x_p = new_A \ new_b

@test ≈(x_p, backslash_x_p, rtol = 1.0e-9)

# Just update A
A, b = h([ForwardDiff.Dual(10.0, 1.0, 0.0), ForwardDiff.Dual(10.0, 0.0, 1.0)])

prob = LinearProblem(A, b)
cache = init(prob, LUFactorization())

new_A, _ = h([ForwardDiff.Dual(5.0, 1.0, 0.0), ForwardDiff.Dual(5.0, 0.0, 1.0)])
cache.A = new_A
@test cache.A == new_A

x_p = solve!(cache)
backslash_x_p = new_A \ b

@test ≈(x_p, backslash_x_p, rtol = 1.0e-9)

# Just update b
A, b = h([ForwardDiff.Dual(5.0, 1.0, 0.0), ForwardDiff.Dual(5.0, 0.0, 1.0)])

prob = LinearProblem(A, b)
cache = init(prob, LUFactorization())

_, new_b = h([ForwardDiff.Dual(5.0, 1.0, 0.0), ForwardDiff.Dual(5.0, 0.0, 1.0)])
cache.b = new_b
@test cache.b == new_b

x_p = solve!(cache)
backslash_x_p = A \ new_b

@test ≈(x_p, backslash_x_p, rtol = 1.0e-9)

# Nested Duals
A,
    b = h(
    [
        ForwardDiff.Dual(ForwardDiff.Dual(5.0, 1.0, 0.0), 1.0, 0.0),
        ForwardDiff.Dual(ForwardDiff.Dual(5.0, 1.0, 0.0), 0.0, 1.0),
    ]
)

prob = LinearProblem(A, b)
overload_x_p = solve(prob)

original_x_p = A \ b

@test ≈(overload_x_p, original_x_p, rtol = 1.0e-9)

prob = LinearProblem(A, b)
cache = init(prob, LUFactorization())

new_A,
    new_b = h(
    [
        ForwardDiff.Dual(ForwardDiff.Dual(10.0, 1.0, 0.0), 1.0, 0.0),
        ForwardDiff.Dual(ForwardDiff.Dual(10.0, 1.0, 0.0), 0.0, 1.0),
    ]
)

cache.A = new_A
cache.b = new_b

@test cache.A == new_A
@test cache.b == new_b

function linprob_f(p)
    A, b = h(p)
    prob = LinearProblem(A, b)
    return solve(prob)
end

function slash_f(p)
    A, b = h(p)
    return A \ b
end

@test ≈(
    ForwardDiff.jacobian(slash_f, [5.0, 5.0]), ForwardDiff.jacobian(linprob_f, [5.0, 5.0])
)

@test ≈(
    ForwardDiff.jacobian(p -> ForwardDiff.jacobian(slash_f, [5.0, p[1]]), [5.0]),
    ForwardDiff.jacobian(p -> ForwardDiff.jacobian(linprob_f, [5.0, p[1]]), [5.0])
)

function g(p)
    return (
        A = [
            p[1] p[1] + 1 p[1]^3;
            3 * p[1] p[1] + 5 p[1] * p[1] - 4;
            p[1]^2 9 * p[1] p[1]
        ],
        b = [p[1] + 1, p[1] * 2, p[1]^2],
    )
end

function slash_f_hes(p)
    A, b = g(p)
    x = A \ b
    return sum(x)
end

function linprob_f_hes(p)
    A, b = g(p)
    prob = LinearProblem(A, b)
    x = solve(prob)
    return sum(x)
end

@test ≈(
    ForwardDiff.hessian(slash_f_hes, [5.0]),
    ForwardDiff.hessian(linprob_f_hes, [5.0])
)

# Test aliasing
A, b = h([ForwardDiff.Dual(5.0, 1.0, 0.0), ForwardDiff.Dual(5.0, 0.0, 1.0)])

prob = LinearProblem(A, b)
cache = init(prob, LUFactorization())

new_A, new_b = h([ForwardDiff.Dual(5.0, 1.0, 0.0), ForwardDiff.Dual(5.0, 0.0, 1.0)])
cache.A = new_A
cache.b = new_b

linu = [
    ForwardDiff.Dual(0.0, 0.0, 0.0), ForwardDiff.Dual(0.0, 0.0, 0.0),
    ForwardDiff.Dual(0.0, 0.0, 0.0),
]
cache.u = linu
x_p = solve!(cache)
backslash_x_p = new_A \ new_b

@test linu == cache.u

# Test Pure Julia Sparse Linear Algebra

A, b = h([ForwardDiff.Dual(5.0, 1.0, 0.0), ForwardDiff.Dual(5.0, 0.0, 1.0)])

prob = LinearProblem(sparse(A), sparse(b))
overload_x_p = solve(prob, PureKLUFactorization())
backslash_x_p = A \ b

@test ≈(overload_x_p, backslash_x_p, rtol = 1.0e-9)

# Duals only in A, and only in b

plain_b = ForwardDiff.value.(b)
prob = LinearProblem(sparse(A), plain_b)
@test ≈(solve(prob, PureKLUFactorization()), A \ plain_b, rtol = 1.0e-9)

plain_A = ForwardDiff.value.(A)
prob = LinearProblem(sparse(plain_A), b)
@test ≈(solve(prob, PureKLUFactorization()), plain_A \ b, rtol = 1.0e-9)

# Mixed-type ldiv!: a primal (Float64) KLU factorization backsolving a Dual RHS
# without promoting A. The factor stays Float64; duals ride through the
# back-substitution (value + each partial column solved in one multi-RHS solve).
@testset "PureKLU primal factor \\ Dual RHS (mixed ldiv!)" begin
    Asp = sparse(2.0I, 5, 5) + sparse(plain_A[1, 1] * 0.0I, 5, 5)
    for i in 1:4
        Asp[i, i + 1] = 0.3
        Asp[i + 1, i] = 0.2
    end
    for nchunk in (1, 2, 3)
        bd = [
            ForwardDiff.Dual{Nothing, Float64, nchunk}(
                    Float64(i), ForwardDiff.Partials(ntuple(k -> sin(i + k), nchunk))
                ) for i in 1:5
        ]
        cache = LinearSolve.__init(LinearProblem(Asp, bd), PureKLUFactorization())
        @test eltype(cache.A) == Float64                 # A not promoted
        u = solve!(cache).u
        uref = Matrix{eltype(bd)}(Asp) \ bd
        @test isapprox(ForwardDiff.value.(u), ForwardDiff.value.(uref); rtol = 1.0e-10)
        @test all(
            isapprox(
                    ForwardDiff.partials(u[i], j), ForwardDiff.partials(uref[i], j);
                    rtol = 1.0e-8, atol = 1.0e-12
                ) for i in 1:5, j in 1:nchunk
        )
    end

    # Duals-only-in-b is routed to a plain LinearCache (native solve), not the split
    # DualLinearCache, and that routing is type-stable.
    Adual, bdual = h([ForwardDiff.Dual(5.0, 1.0, 0.0), ForwardDiff.Dual(5.0, 0.0, 1.0)])
    plain_Asp = sparse(ForwardDiff.value.(Adual))
    bprob = LinearProblem(plain_Asp, bdual)
    @test init(bprob, PureKLUFactorization()) isa LinearSolve.LinearCache
    @test (@inferred init(bprob, PureKLUFactorization())) isa LinearSolve.LinearCache
    @test ≈(solve(bprob, PureKLUFactorization()), Adual \ bdual, rtol = 1.0e-9)
end

A, b = h([ForwardDiff.Dual(5.0, 1.0, 0.0), ForwardDiff.Dual(5.0, 0.0, 1.0)])

prob = LinearProblem(sparse(A), sparse(b))
overload_x_p = solve(prob, SparseColumnPivotedQRFactorization())
backslash_x_p = A \ b

@test ≈(overload_x_p, backslash_x_p, rtol = 1.0e-9)

# Test Float Only solvers

A, b = h([ForwardDiff.Dual(5.0, 1.0, 0.0), ForwardDiff.Dual(5.0, 0.0, 1.0)])

prob = LinearProblem(sparse(A), sparse(b))
overload_x_p = solve(prob, KLUFactorization())
backslash_x_p = A \ b

@test ≈(overload_x_p, backslash_x_p, rtol = 1.0e-9)

A, b = h([ForwardDiff.Dual(5.0, 1.0, 0.0), ForwardDiff.Dual(5.0, 0.0, 1.0)])

prob = LinearProblem(sparse(A), sparse(b))
cache = init(prob, UMFPACKFactorization())
overload_x_p = solve!(cache)
backslash_x_p = A \ b

@test ≈(overload_x_p, backslash_x_p, rtol = 1.0e-9)

A[1, 1] += 2
reinit!(cache; A = sparse(A))
overload_x_p = solve!(cache, UMFPACKFactorization())
backslash_x_p = A \ b
@test ≈(overload_x_p, backslash_x_p, rtol = 1.0e-9)

# Test type inference for init with ForwardDiff Dual numbers
# This ensures init returns a concrete type (not a Union) for type stability
A, b = h([ForwardDiff.Dual(5.0, 1.0, 0.0), ForwardDiff.Dual(5.0, 0.0, 1.0)])

prob = LinearProblem(A, b)

# Helper to check if type is DualLinearCache (extension type not directly accessible)
is_dual_cache(x) = nameof(typeof(x)) == :DualLinearCache

# GenericLUFactorization now returns DualLinearCache for type stability
# (the optimization for GenericLU happens at solve-time instead of init-time)
@test is_dual_cache(init(prob, GenericLUFactorization()))

# Test inference with explicit algorithm
@test is_dual_cache(@inferred init(prob, LUFactorization()))
@test is_dual_cache(@inferred init(prob, GenericLUFactorization()))

# Test inference with default algorithm (nothing) - this was the main bug
# Previously returned Union{LinearCache, DualLinearCache} due to runtime conditional
@test is_dual_cache(@inferred init(prob, nothing))

# Test that SparspakFactorization still opts out (sparse solvers can't handle Duals the same way)
A, b = h([ForwardDiff.Dual(5.0, 1.0, 0.0), ForwardDiff.Dual(5.0, 0.0, 1.0)])

prob = LinearProblem(sparse(A), b)
@test init(prob, SparspakFactorization()) isa LinearSolve.LinearCache

# Test that solve still works correctly with GenericLUFactorization
A, b = h([ForwardDiff.Dual(5.0, 1.0, 0.0), ForwardDiff.Dual(5.0, 0.0, 1.0)])
prob = LinearProblem(A, b)
sol_generic = solve(prob, GenericLUFactorization())
backslash_result = A \ b
@test ≈(sol_generic.u, backslash_result, rtol = 1.0e-9)

# Test ComponentArray with ForwardDiff (Issue SciML/DifferentialEquations.jl#1110)
# This tests that ArrayInterface.restructure preserves ComponentArray structure

# Direct test: ComponentVector with Dual elements should preserve structure
ca_dual = ComponentArray(
    a = ForwardDiff.Dual(1.0, 1.0, 0.0),
    b = ForwardDiff.Dual(2.0, 0.0, 1.0)
)
A_dual = [ca_dual.a 1.0; 1.0 ca_dual.b]
b_dual = ComponentArray(x = ca_dual.a + 1, y = ca_dual.b * 2)

prob_dual = LinearProblem(A_dual, b_dual)
sol_dual = solve(prob_dual)

# The solution should preserve ComponentArray type
@test sol_dual.u isa ComponentVector
@test hasproperty(sol_dual.u, :x)
@test hasproperty(sol_dual.u, :y)

# Test gradient computation with ComponentArray inside ForwardDiff
function component_linsolve(p)
    # Create a matrix that depends on p
    A = [p[1] p[2]; p[2] p[1] + 5]
    # Create a ComponentArray RHS that depends on p
    b_vec = ComponentArray(x = p[1] + 1, y = p[2] * 2)
    prob = LinearProblem(A, b_vec)
    sol = solve(prob)
    # Return sum of solution
    return sum(sol.u)
end

p_test = [2.0, 3.0]
# This will internally create Dual numbers and ComponentArrays with Dual elements
grad = ForwardDiff.gradient(component_linsolve, p_test)
@test grad isa Vector
@test length(grad) == 2
@test !any(isnan, grad)
@test !any(isinf, grad)

# Test overdetermined (non-square) system: 2×1 matrix with dual numbers
# This tests that cache sizes are correctly allocated when solution size != RHS size
A_overdet = reshape([ForwardDiff.Dual(2.0, 1.0), ForwardDiff.Dual(3.0, 1.0)], 2, 1)  # 2×1 matrix
b_overdet = [ForwardDiff.Dual(5.0, 1.0), ForwardDiff.Dual(8.0, 9.0)]

prob_overdet = LinearProblem(A_overdet, b_overdet)
sol_overdet = solve(prob_overdet)
backslash_overdet = A_overdet \ b_overdet

# Test that solution has correct dimensions (length 1, not length 2)
@test length(sol_overdet.u) == 1

# Primal values should match
@test ForwardDiff.value.(sol_overdet.u) ≈ ForwardDiff.value.(backslash_overdet)

# Dual values should match
@test ForwardDiff.partials.(sol_overdet.u) ≈ ForwardDiff.partials.(backslash_overdet)

# Test with cache - should give identical results
cache_overdet = init(prob_overdet)
sol_cache_overdet = solve!(cache_overdet)
@test sol_cache_overdet.u ≈ sol_overdet.u

# Dual values should match
@test ForwardDiff.partials.(sol_overdet.u) ≈ ForwardDiff.partials.(backslash_overdet)

# Test larger overdetermined system with dual numbers
m, n = 10, 3
A_large = rand(m, n)
p = [2.0, 3.0]
A_large_dual = [
    ForwardDiff.Dual(A_large[i, j], i == 1 ? 1.0 : 0.0, j == 1 ? 1.0 : 0.0)
        for i in 1:m, j in 1:n
]
b_large_dual = [
    ForwardDiff.Dual(rand(), i == 1 ? 1.0 : 0.0, i == 2 ? 1.0 : 0.0)
        for i in 1:m
]

prob_large = LinearProblem(A_large_dual, b_large_dual)
sol_large = solve(prob_large)
backslash_large = A_large_dual \ b_large_dual

# Test primal values match
@test ForwardDiff.value.(sol_large.u) ≈ ForwardDiff.value.(backslash_large)

@test A_large_dual' * A_large_dual * sol_large.u ≈ A_large_dual' * b_large_dual
@test A_large_dual' * A_large_dual * backslash_large ≈ A_large_dual' * b_large_dual

# Test partials match
@test ForwardDiff.partials.(sol_large.u) ≈ ForwardDiff.partials.(backslash_large)

# Test that DualLinearCache preserves p through init and reinit!
# This is needed by OrdinaryDiffEq which passes ODE state as p to LinearProblem
# for preconditioner access.
@testset "DualLinearCache preserves p parameter" begin
    function solve_with_p(params)
        A = [params[1] 1.0; 0.0 params[2]]
        b = [params[1] + 1.0, params[2] * 2.0]
        p = (nothing, params, 0.0)
        prob = LinearProblem(A, b, p)
        cache = init(prob, nothing)
        sol = solve!(cache)
        # p on DualLinearCache returns de-dualed values from inner cache
        @test cache.p == (nothing, ForwardDiff.value.(params), 0.0)
        return sum(sol.u)
    end

    # Test that ForwardDiff can differentiate through solve with p
    grad = ForwardDiff.gradient(solve_with_p, [2.0, 3.0])
    @test length(grad) == 2

    # Test reinit! with new p value on DualLinearCache
    A_dual, b_dual = h([ForwardDiff.Dual(5.0, 1.0, 0.0), ForwardDiff.Dual(5.0, 0.0, 1.0)])
    p_init = (nothing, [1.0, 2.0], 0.0)
    prob_p = LinearProblem(A_dual, b_dual, p_init)
    cache_p = init(prob_p, nothing)
    sol1 = solve!(cache_p)

    # reinit! with a new p should not error
    new_p = (nothing, [3.0, 4.0], 1.0)
    @test_nowarn LinearSolve.reinit!(cache_p; A = A_dual, p = new_p)
    @test cache_p.p == new_p

    # setproperty! for p should also work
    another_p = (nothing, [5.0, 6.0], 2.0)
    @test_nowarn (cache_p.p = another_p)
    @test cache_p.p == another_p
end

# Regression test for SciML/LinearSolve.jl#972
# When DualLinearCache is built from a Dual A and a Float64 b (so partials_b
# is `nothing`), reusing the cache by mutating b to another Float64 vector
# previously hit `MethodError: no method matching map!(::partial_vals,
# ::Nothing, ::Vector{Float64})` from `setb!`. The same applied symmetrically
# to setA!/setu! when their partials slots were unallocated.
@testset "DualLinearCache reuse with nothing partials (#972)" begin
    function f972(p)
        A = sparse(Diagonal(p))           # A is Dual under ForwardDiff
        b1 = [1.0, 2.0]                   # Float64, no partials
        prob = LinearProblem(A, b1)
        cache = init(prob)
        u1 = copy(solve!(cache).u)        # first solve seeded the cache
        cache.b = [3.0, 4.0]              # mutating b to another Float64 vector used to MethodError
        u2 = copy(solve!(cache).u)
        return u1 .+ u2
    end

    # Primal call must still work and produce a sensible result.
    @test f972([1.0, 2.0]) ≈ [4.0, 3.0]

    # The original failure mode: pushing Duals through the same cache path.
    J = ForwardDiff.jacobian(f972, [1.0, 2.0])
    @test size(J) == (2, 2)
    @test all(isfinite, J)

    # Sanity-check the partials by comparing against a non-cached implementation.
    function f972_nocache(p)
        A = sparse(Diagonal(p))
        u1 = A \ [1.0, 2.0]
        u2 = A \ [3.0, 4.0]
        return u1 .+ u2
    end
    @test J ≈ ForwardDiff.jacobian(f972_nocache, [1.0, 2.0])

    # Symmetric coverage: Float64 A + Dual b cache reuse via setA!.
    function f972_setA(p)
        A1 = Matrix{Float64}(I, 2, 2)
        b = p                              # b is Dual
        prob = LinearProblem(A1, b)
        cache = init(prob)
        u1 = copy(solve!(cache).u)
        cache.A = [2.0 0.0; 0.0 2.0]       # mutate A to another Float64 matrix
        u2 = copy(solve!(cache).u)
        return u1 .+ u2
    end
    @test ForwardDiff.jacobian(f972_setA, [1.0, 2.0]) ≈
        [1.5 0.0; 0.0 1.5]
end

# Regression test for SciML/LinearSolve.jl#974
# When a DualLinearCache is constructed with a Dual A and Float64 b, the
# `dual_u` field is statically typed Vector{<:Dual}. NonlinearSolveBase's
# `set_lincache_u!` (which does `cache.u = primal_u_vector`) then routes
# through `setproperty!(dc, :u, ::Vector{Float64})` → `setu!` →
# `setfield!(dc, :dual_u, ::Vector{Float64})`, which TypeError'd on the
# typed field. Hit in practice by NonlinearSolve.NewtonRaphson under
# ForwardDiff.hessian (outer Dual tag specializes dual_u against an
# outer-Dual type while the inner solve's iterate is Float64). The fix
# promotes the primal-only `u` to the cache's Dual type with zero partials
# so the field invariant is preserved.
@testset "DualLinearCache setu! with non-Dual u (#974)" begin
    p = [ForwardDiff.Dual(1.0, 1.0, 0.0), ForwardDiff.Dual(2.0, 0.0, 1.0)]
    A = sparse(Diagonal(p))                # sparse Dual A
    b = [1.0, 2.0]                         # Float64 b (no partials)
    prob = LinearProblem(A, b)
    cache = init(prob)
    solve!(cache)
    DT = eltype(getfield(cache, :dual_u))
    @test DT <: ForwardDiff.Dual

    # On master this errors with `TypeError: in setfield!, expected
    # Vector{ForwardDiff.Dual{...}}, got a value of type Vector{Float64}`.
    new_u = [0.5, 1.5]
    @test_nowarn (cache.u = new_u)
    @test cache.linear_cache.u == new_u

    # The dual_u field invariant must be preserved: still Vector{DT}, with
    # primal values matching `new_u` and zero partials. Derivatives must not
    # be dropped — the next solve! will rewrite the partials from `A`.
    promoted = getfield(cache, :dual_u)
    @test eltype(promoted) === DT
    @test ForwardDiff.value.(promoted) == new_u
    @test all(iszero, ForwardDiff.partials.(promoted))

    # A subsequent solve must still produce the correct dual solution —
    # values AND partials. `≈` on Dual only compares values, so check the
    # extracted partials matrix against `Diagonal(p) \ b` separately.
    sol = solve!(cache)
    ref = Diagonal(p) \ b
    @test eltype(sol.u) === DT
    @test ForwardDiff.value.(sol.u) ≈ ForwardDiff.value.(ref) rtol = 1.0e-9
    extract_partials(v) = [collect(ForwardDiff.partials(x)) for x in v]
    @test all(((a, b),) -> a ≈ b, zip(extract_partials(sol.u), extract_partials(ref)))
end

# The DualLinearCache tracks partials-list validity for A and b independently,
# so mutating only one side does not force the other's partials to be recomputed
# (relevant e.g. in an ODE where A is fixed while b changes, and vice versa).
@testset "DualLinearCache separate A/b partials validity" begin
    A, b = h([ForwardDiff.Dual(10.0, 1.0, 0.0), ForwardDiff.Dual(10.0, 0.0, 1.0)])
    cache = init(LinearProblem(A, b), LUFactorization())

    # Both lists start valid (populated lazily on first solve).
    @test getfield(cache, :A_partials_valid)
    @test getfield(cache, :b_partials_valid)

    # Mutating only b invalidates b's list and leaves A's untouched.
    _, new_b = h([ForwardDiff.Dual(5.0, 1.0, 0.0), ForwardDiff.Dual(5.0, 0.0, 1.0)])
    cache.b = new_b
    @test getfield(cache, :A_partials_valid)
    @test !getfield(cache, :b_partials_valid)

    # Solving revalidates both, and the result still matches the reference.
    x_p = solve!(cache)
    @test getfield(cache, :A_partials_valid)
    @test getfield(cache, :b_partials_valid)
    @test ≈(x_p, A \ new_b, rtol = 1.0e-9)

    # Symmetrically, mutating only A invalidates A's list and leaves b's untouched.
    new_A, _ = h([ForwardDiff.Dual(2.0, 1.0, 0.0), ForwardDiff.Dual(2.0, 0.0, 1.0)])
    cache.A = new_A
    @test !getfield(cache, :A_partials_valid)
    @test getfield(cache, :b_partials_valid)

    x_p = solve!(cache)
    @test getfield(cache, :A_partials_valid)
    @test getfield(cache, :b_partials_valid)
    @test ≈(x_p, new_A \ new_b, rtol = 1.0e-9)
end

@testset "Sparse matrices" begin
    # Case 1: same number of nonzeros, same sparsity pattern, updated values
    A1 = sparse([1, 2], [1, 2], [ForwardDiff.Dual(1.0, 10.0, 11.0), ForwardDiff.Dual(2.0, 20.0, 21.0)], 2, 2)
    b = [1.0, 1.0]
    prob = LinearProblem(A1, b)
    cache = init(prob, KLUFactorization())
    solve!(cache)
    cache.A = A1
    @test ≈(solve!(cache), Matrix(A1) \ b, rtol = 1.0e-9)

    # Case 2: same number of nonzeros, but different positions
    A2 = sparse([2, 1], [1, 2], [ForwardDiff.Dual(3.0, 30.0, 31.0), ForwardDiff.Dual(4.0, 40.0, 41.0)], 2, 2)
    cache.A = A2
    @test ≈(solve!(cache), Matrix(A2) \ b, rtol = 1.0e-9)

    # Case 3: number of nonzeros increases
    A3 = sparse([1, 2, 1], [1, 2, 2], [ForwardDiff.Dual(3.0, 30.0, 31.0), ForwardDiff.Dual(4.0, 40.0, 41.0), ForwardDiff.Dual(5.0, 50.0, 51.0)], 2, 2)
    cache.A = A3
    @test ≈(solve!(cache), Matrix(A3) \ b, rtol = 1.0e-9)

    # Case 4: dual b (both A and b carry partials)
    b_dual = [ForwardDiff.Dual(3.0, 1.0, 0.0), ForwardDiff.Dual(4.0, 0.0, 1.0)]
    prob = LinearProblem(A1, b_dual)
    cache = init(prob, KLUFactorization())
    @test ≈(solve!(cache), Matrix(A1) \ b_dual, rtol = 1.0e-9)
    b_dual2 = [ForwardDiff.Dual(5.0, 2.0, 0.0), ForwardDiff.Dual(6.0, 0.0, 2.0)]
    cache.b = b_dual2
    @test ≈(solve!(cache), Matrix(A1) \ b_dual2, rtol = 1.0e-9)

    # Case 5: in-place mutation of A's nonzeros via setindex! (ODE solver pattern)
    A = sparse([1, 2], [1, 2], [ForwardDiff.Dual(1.0, 10.0, 11.0), ForwardDiff.Dual(2.0, 20.0, 21.0)], 2, 2)
    prob = LinearProblem(A, b)
    cache = init(prob, KLUFactorization())
    solve!(cache)
    A[1, 1] = ForwardDiff.Dual(5.0, 50.0, 51.0)
    A[2, 2] = ForwardDiff.Dual(6.0, 60.0, 61.0)
    cache.A = A
    @test ≈(solve!(cache), Matrix(A) \ b, rtol = 1.0e-9)
end

@testset "PureKLU direct dual path reuses inner LinearCache" begin
    # The direct dual path (PureKLUFactorization can handle Dual arithmetic
    # natively) pre-creates an inner LinearCache in DualLinearCache.dual_linear_cache
    # at init time and reuses it on every solve!, only re-factorising when
    # linear_cache.isfresh is set. Verify that the inner cache object identity
    # is preserved across successive solves with different A and b.
    A1 = sparse([1, 2], [1, 2], [ForwardDiff.Dual(1.0, 10.0, 0.0), ForwardDiff.Dual(2.0, 0.0, 20.0)], 2, 2)
    b1 = [ForwardDiff.Dual(3.0, 1.0, 0.0), ForwardDiff.Dual(4.0, 0.0, 1.0)]
    prob = LinearProblem(A1, b1)
    cache = init(prob, PureKLUFactorization())

    # The inner dual cache must be pre-created at init time (not nothing).
    inner = getfield(cache, :dual_linear_cache)
    @test inner !== nothing

    sol1 = solve!(cache)
    @test ≈(sol1, Matrix(A1) \ b1, rtol = 1.0e-9)

    # After mutating A the inner cache object must be the same — only its
    # contents change, no new allocation.
    A2 = sparse([1, 2], [1, 2], [ForwardDiff.Dual(3.0, 30.0, 0.0), ForwardDiff.Dual(4.0, 0.0, 40.0)], 2, 2)
    cache.A = A2
    sol2 = solve!(cache)
    @test getfield(cache, :dual_linear_cache) === inner
    @test ≈(sol2, Matrix(A2) \ b1, rtol = 1.0e-9)

    # Same after mutating only b.
    b2 = [ForwardDiff.Dual(5.0, 2.0, 0.0), ForwardDiff.Dual(6.0, 0.0, 2.0)]
    cache.b = b2
    sol3 = solve!(cache)
    @test getfield(cache, :dual_linear_cache) === inner
    @test ≈(sol3, Matrix(A2) \ b2, rtol = 1.0e-9)
end
