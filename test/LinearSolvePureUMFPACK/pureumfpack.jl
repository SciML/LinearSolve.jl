using LinearSolve
using PureUMFPACK  # plain `using` must not shadow `solve` (PureUMFPACK >= 1.0)
using SparseArrays
using LinearAlgebra
using Test

@test Base.get_extension(LinearSolve, :LinearSolvePureUMFPACKExt) !== nothing

n = 40
A = sprand(n, n, 0.2) + 5I
b = rand(n)

@testset "PureUMFPACKFactorization: square solve" begin
    prob = LinearProblem(A, b)
    sol = solve(prob, PureUMFPACKFactorization())
    @test sol.retcode == ReturnCode.Success
    @test norm(A * sol.u - b) < 1.0e-9
end

@testset "PureUMFPACKFactorization: caching / refactor" begin
    prob = LinearProblem(A, b)
    cache = init(prob, PureUMFPACKFactorization())
    sol1 = solve!(cache)
    @test norm(A * sol1.u - b) < 1.0e-9

    # New rhs, same factorization (cached) -> reuse
    b2 = rand(n)
    cache.b = b2
    sol2 = solve!(cache)
    @test norm(A * sol2.u - b2) < 1.0e-9

    # New matrix values, same sparsity pattern -> refactor
    A2 = copy(A)
    A2.nzval .*= 2.5
    cache.A = A2
    sol3 = solve!(cache)
    @test norm(A2 * sol3.u - b2) < 1.0e-9
end

@testset "PureUMFPACKFactorization: reuse_symbolic = false" begin
    prob = LinearProblem(A, b)
    sol = solve(prob, PureUMFPACKFactorization(reuse_symbolic = false))
    @test sol.retcode == ReturnCode.Success
    @test norm(A * sol.u - b) < 1.0e-9
end

@testset "PureUMFPACKFactorization: errors when PureUMFPACK is not loaded" begin
    # Without the extension the algorithm would otherwise fall through to the generic
    # `AbstractFactorization` `solve!` and hit a `do_factorization` MethodError.
    script = """
    using LinearSolve
    @assert Base.get_extension(LinearSolve, :LinearSolvePureUMFPACKExt) === nothing
    try
        PureUMFPACKFactorization()
        exit(1)
    catch e
        e isa ErrorException && occursin("PureUMFPACK", e.msg) || exit(2)
    end
    PureUMFPACKFactorization(throwerror = false)
    exit(0)
    """
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -e $script`
    @test success(run(cmd; wait = true))
end

@testset "PureUMFPACKFactorization: singular -> Infeasible" begin
    As = sparse([1, 2, 1, 2], [1, 1, 2, 2], [1.0, 2.0, 2.0, 4.0], 2, 2)
    prob = LinearProblem(As, [1.0, 2.0])
    sol = solve(prob, PureUMFPACKFactorization())
    @test sol.retcode == ReturnCode.Infeasible
end
