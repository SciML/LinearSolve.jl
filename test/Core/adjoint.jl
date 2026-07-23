using Zygote, ForwardDiff
using LinearSolve, LinearAlgebra, Test
using FiniteDiff, RecursiveFactorization
using InteractiveUtils, Random, SparseArrays
import CliqueTrees

struct UnregisteredFactorization <: LinearSolve.AbstractFactorization end

function factorization_leaf_types(T)
    children = subtypes(T)
    isempty(children) && return Any[T]
    return reduce(vcat, factorization_leaf_types.(children); init = Any[])
end

if Sys.islinux()
    import LAPACK_jll, blis_jll
end

Random.seed!(1234)
n = 4
A = rand(n, n);
b1 = rand(n);

@testset "Adjoint factorization cache dispatch follows the solver" begin
    factorization = lu(copy(A))
    @test LinearSolve._cache_factorization(LUFactorization(), factorization) ===
        factorization
    @test LinearSolve._cache_factorization(
        GenericLUFactorization(), (factorization, factorization.ipiv)
    ) === factorization
    @test LinearSolve._can_reuse_cache_factorization(
        LUFactorization(), factorization
    )

    krylov = KrylovJL_GMRES()
    @test isnothing(LinearSolve._cache_factorization(krylov, factorization))
    @test !LinearSolve._can_reuse_cache_factorization(krylov, factorization)

    default = LinearSolve.DefaultLinearSolver(
        LinearSolve.DefaultAlgorithmChoice.LUFactorization
    )
    @test isnothing(LinearSolve._cache_factorization(default, factorization))
    @test !LinearSolve._can_reuse_cache_factorization(default, factorization)

    unregistered = UnregisteredFactorization()
    @test isnothing(LinearSolve._cache_factorization(unregistered, factorization))
    @test !LinearSolve._can_reuse_cache_factorization(unregistered, factorization)
end

@testset "Complex Krylov adjoint solve" begin
    A_complex = ComplexF64[3 + 1im 1 - 2im; 2 + 0.5im 4 - 1im]
    rhs_complex = ComplexF64[0.7 + 0.2im, -0.3 + 0.4im]
    adjoint_solution = LinearSolve._adjoint_krylov_solve(
        KrylovJL_GMRES(), A_complex, rhs_complex;
        abstol = 1.0e-12, reltol = 1.0e-12, verbose = false
    )
    @test adjoint(A_complex) * adjoint_solution ≈ rhs_complex
end

@testset "Every factorization algorithm declares its adjoint reuse policy" begin
    for T in factorization_leaf_types(LinearSolve.AbstractFactorization)
        parentmodule(T) === LinearSolve || continue
        reuse = LinearSolve._adjoint_factorization_reuse(T)
        @test !(reuse isa LinearSolve._UnspecifiedAdjointFactorizationReuse)
    end
end

@testset "Solver-specific cached adjoint solves" begin
    A_local = [4.0 1.0; 2.0 3.0]
    b_local = [1.0, 2.0]
    adjoint_rhs = [0.7, -0.3]

    for alg in (
            NormalCholeskyFactorization(),
            NormalBunchKaufmanFactorization(),
            SimpleLUFactorization(),
        )
        cache = init(LinearProblem(copy(A_local), copy(b_local)), alg)
        @test LinearSolve._can_reuse_cache_factorization(alg, cache.cacheval)
        solve!(cache)
        adjoint_solution = LinearSolve._adjoint_factorization_solve(
            alg, cache.cacheval, cache.A, adjoint_rhs
        )
        @test adjoint(A_local) * adjoint_solution ≈ adjoint_rhs
    end

    tall_A = [4.0 1.0; 2.0 3.0; 1.0 -1.0]
    tall_b = [1.0, 2.0, -0.5]
    tall_adjoint_rhs = [0.7, -0.3]
    for alg in (NormalCholeskyFactorization(), NormalBunchKaufmanFactorization())
        cache = init(LinearProblem(copy(tall_A), copy(tall_b)), alg)
        solve!(cache)
        adjoint_solution = LinearSolve._adjoint_factorization_solve(
            alg, cache.cacheval, cache.A, tall_adjoint_rhs
        )
        @test adjoint_solution ≈ tall_A * ((adjoint(tall_A) * tall_A) \ tall_adjoint_rhs)
        @test adjoint(tall_A) * adjoint_solution ≈ tall_adjoint_rhs
    end

    sparse_A = sparse(A_local)
    sparse_alg = SparseColumnPivotedQRFactorization()
    sparse_cache = init(LinearProblem(copy(sparse_A), copy(b_local)), sparse_alg)
    @test LinearSolve._can_reuse_cache_factorization(
        sparse_alg, sparse_cache.cacheval
    )
    solve!(sparse_cache)
    sparse_adjoint_solution = LinearSolve._adjoint_factorization_solve(
        sparse_alg, sparse_cache.cacheval, sparse_cache.A, adjoint_rhs
    )
    @test adjoint(sparse_A) * sparse_adjoint_solution ≈ adjoint_rhs

    clique_A = sparse([4.0 1.0; 1.0 3.0])
    clique_alg = CliqueTreesFactorization()
    clique_cache = init(LinearProblem(copy(clique_A), copy(b_local)), clique_alg)
    @test LinearSolve._can_reuse_cache_factorization(
        clique_alg, clique_cache.cacheval
    )
    solve!(clique_cache)
    clique_adjoint_solution = LinearSolve._adjoint_factorization_solve(
        clique_alg, clique_cache.cacheval, clique_cache.A, adjoint_rhs
    )
    @test adjoint(clique_A) * clique_adjoint_solution ≈ adjoint_rhs

    for alg in (NormalCholeskyFactorization(), SimpleLUFactorization(), sparse_alg)
        A_alg = alg isa SparseColumnPivotedQRFactorization ? sparse_A : A_local
        db, = Zygote.gradient(b -> sum(solve(LinearProblem(A_alg, b), alg).u), b_local)
        @test db ≈ adjoint(A_local) \ ones(2)
    end
end

@testset "Uncached factorization adjoint fallback" begin
    diagonal = rand(n) .+ 1
    A_diagonal = Diagonal(diagonal)
    f_diagonal(b) = sum(
        solve(LinearProblem(A_diagonal, b), DiagonalFactorization()).u
    )
    db, = Zygote.gradient(f_diagonal, b1)
    @test db ≈ inv.(diagonal)
end

function f(A, b1; alg = LUFactorization())
    prob = LinearProblem(A, b1)

    sol1 = solve(prob, alg)

    s1 = sol1.u
    return norm(s1)
end

f(A, b1) # Uses BLAS

dA, db1 = Zygote.gradient(f, A, b1)
@test dA isa AbstractMatrix

dA2 = ForwardDiff.gradient(x -> f(x, eltype(x).(b1)), copy(A))
db12 = ForwardDiff.gradient(x -> f(eltype(x).(A), x), copy(b1))

@test dA ≈ dA2
@test db1 ≈ db12

A = rand(n, n);
b1 = rand(n);

_ff = (
    x,
    y,
) -> f(
    x,
    y;
    alg = LinearSolve.DefaultLinearSolver(LinearSolve.DefaultAlgorithmChoice.LUFactorization)
)
_ff(copy(A), copy(b1))

dA, db1 = Zygote.gradient(_ff, copy(A), copy(b1))
@test dA isa AbstractMatrix

dA2 = ForwardDiff.gradient(x -> f(x, eltype(x).(b1)), copy(A))
db12 = ForwardDiff.gradient(x -> f(eltype(x).(A), x), copy(b1))

@test dA ≈ dA2
@test db1 ≈ db12

# Test complex numbers
A = rand(n, n) + 1im * rand(n, n);
b1 = rand(n) + 1im * rand(n);

function f3(A, b1, b2; alg = KrylovJL_GMRES())
    prob = LinearProblem(A, b1)
    sol1 = solve(prob, alg)
    prob = LinearProblem(A, b2)
    sol2 = solve(prob, alg)
    return norm(sol1.u .+ sol2.u)
end

dA, db1, db2 = Zygote.gradient(f3, A, b1, b1)
@test dA isa AbstractMatrix

dA2 = FiniteDiff.finite_difference_gradient(
    x -> f3(x, eltype(x).(b1), eltype(x).(b1)), copy(A)
)
db12 = FiniteDiff.finite_difference_gradient(
    x -> f3(eltype(x).(A), x, eltype(x).(b1)), copy(b1)
)
db22 = FiniteDiff.finite_difference_gradient(
    x -> f3(eltype(x).(A), eltype(x).(b1), x), copy(b1)
)

@test dA ≈ dA2 rtol = 1.0e-3
@test db1 ≈ db12
@test db2 ≈ db22

A = rand(n, n);
b1 = rand(n);

function f4(A, b1, b2; alg = LUFactorization())
    prob = LinearProblem(A, b1)
    sol1 = solve(prob, alg; sensealg = LinearSolveAdjoint(; linsolve = KrylovJL_LSMR()))
    prob = LinearProblem(A, b2)
    sol2 = solve(prob, alg; sensealg = LinearSolveAdjoint(; linsolve = KrylovJL_GMRES()))
    return norm(sol1.u .+ sol2.u)
end

dA, db1, db2 = Zygote.gradient(f4, A, b1, b1)
@test dA isa AbstractMatrix

dA2 = ForwardDiff.gradient(x -> f4(x, eltype(x).(b1), eltype(x).(b1)), copy(A))
db12 = ForwardDiff.gradient(x -> f4(eltype(x).(A), x, eltype(x).(b1)), copy(b1))
db22 = ForwardDiff.gradient(x -> f4(eltype(x).(A), eltype(x).(b1), x), copy(b1))

@test dA ≈ dA2 atol = 5.0e-5
@test db1 ≈ db12
@test db2 ≈ db22

A = rand(n, n);
b1 = rand(n);
adjoint_algs = Any[
    LUFactorization(),
    RFLUFactorization(),
    KrylovJL_GMRES(),
]
LinearSolve.useopenblas && push!(adjoint_algs, OpenBLASLUFactorization())
if Base.get_extension(LinearSolve, :LinearSolveBLISExt) !== nothing
    push!(adjoint_algs, LinearSolve.BLISLUFactorization())
end
for alg in adjoint_algs
    @show alg
    function fb(b)
        prob = LinearProblem(A, b)

        sol1 = solve(prob, alg)

        return sum(sol1.u)
    end
    fb(b1)

    fd_jac = FiniteDiff.finite_difference_jacobian(fb, b1) |> vec
    @show fd_jac

    zyg_jac = Zygote.jacobian(fb, b1) |> first |> vec
    @show zyg_jac

    @test zyg_jac ≈ fd_jac rtol = 5.0e-4

    function fA(A)
        prob = LinearProblem(A, b1)

        sol1 = solve(prob, alg)

        return sum(sol1.u)
    end
    fA(A)

    fd_jac = FiniteDiff.finite_difference_jacobian(fA, A) |> vec
    @show fd_jac

    zyg_jac = Zygote.jacobian(fA, A) |> first |> vec
    @show zyg_jac

    @test zyg_jac ≈ fd_jac rtol = 5.0e-4
end

@testset "Direct solution indexing without .u" begin
    N = 2
    Random.seed!(1234)
    function test_func(x::AbstractVector{T}) where {T <: Real}
        A = reshape(x[1:(N * N)], (N, N))
        b = x[(N * N + 1):end]
        prob = LinearProblem(A, b)
        sol = solve(prob)
        return sum(sol)
    end

    x0 = rand(N * N + N)

    grad_zygote = Zygote.gradient(test_func, x0)
    grad_forwarddiff = ForwardDiff.gradient(test_func, x0)
    @test grad_zygote[1] ≈ grad_forwarddiff rtol = 1.0e-5
end

struct System end
function update_A!(A, p::Vector{T}) where {T}
    A[1, 1] = sum(p)
    A[1, 2] = one(T)
    A[2, 1] = 2one(T)
    return A[2, 2] = 3one(T)
end
function update_b!(b, p::Vector{T}) where {T}
    b[1] = p[1]
    return b[2] = 2one(T)
end

function SciMLBase.get_new_A_b(::System, f, p, A, b; kw...)
    if eltype(A) != eltype(p)
        A = similar(A, eltype(p))
        b = similar(b, eltype(p))
    end
    f.update_A!(A, p)
    f.update_b!(b, p)
    return A, b
end


@testset "Avoid stackoverflow with `SciMLBase.get_new_A_b` hook" begin
    # See https://github.com/SciML/LinearSolve.jl/pull/868#issuecomment-3723338914
    p = [1.0, 2.0, 3.0]
    A = [6.0 1.0; 2.0 3.0]
    b = [1.0, 2.0]
    linfun = SciMLBase.SymbolicLinearInterface(update_A!, update_b!, System(), nothing, nothing)

    prob = LinearProblem(A, b; f = linfun)
    sol = solve(prob)

    newp = [2.0, 3.0, 4.0]
    @test_nowarn ForwardDiff.gradient(newp) do p
        prob2 = remake(prob; p)
        sum(solve(prob2).u)
    end
end
