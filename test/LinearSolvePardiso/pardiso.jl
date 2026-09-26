# Julia 1.13's SparseArrays stdlib loads libgomp, which makes MKL select
# libmkl_gnu_thread; its complex CGS path (used by MKLPardisoIterate) crashes or
# returns zeros there. MKL must bind a threading layer at first use, before
# `import Pardiso` triggers initialization, and refuses the intel layer while
# libgomp is loaded, so run the suite on the sequential layer.
get(ENV, "MKL_THREADING_LAYER", "") == "" && (ENV["MKL_THREADING_LAYER"] = "sequential")

using LinearSolve, SparseArrays, Random, LinearAlgebra
using Test
import Pardiso

Random.seed!(1234)

A1 = sparse(
    [
        1.0 0 -2 3
        0 5 1 2
        -2 1 4 -7
        3 2 -7 5
    ]
)
b1 = rand(4)
prob1 = LinearProblem(A1, b1)

lambda = 3
n = 4
e = ones(n)
e2 = ones(n - 1)
A2 = spdiagm(-1 => 1.0 .+ im * e2, 0 => lambda * e, 1 => 1.0 .+ -im * e2)
b2 = rand(n) + im * zeros(n)
cache_kwargs = (; abstol = 1.0e-8, reltol = 1.0e-8, maxiter = 30)

prob2 = LinearProblem(A2, b2)

algs = LinearSolve.SciMLLinearSolveAlgorithm[PardisoJL()]
solvers = Pardiso.AbstractPardisoSolver[]
extended_algs = LinearSolve.SciMLLinearSolveAlgorithm[PardisoJL()]

if Pardiso.mkl_is_available()
    push!(algs, MKLPardisoFactorize())
    push!(solvers, Pardiso.MKLPardisoSolver())
    extended_algs = vcat(extended_algs, [MKLPardisoFactorize(), MKLPardisoIterate()])
    @info "Testing MKL Pardiso"
end

if Pardiso.panua_is_available()
    push!(algs, PanuaPardisoFactorize())
    push!(solvers, Pardiso.PardisoSolver())
    extended_algs = vcat(extended_algs, [PanuaPardisoFactorize(), PanuaPardisoIterate()])
    @info "Testing Panua Pardiso"
end

for alg in extended_algs
    u = solve(prob1, alg; cache_kwargs...).u
    @test A1 * u ≈ b1

    u = solve(prob2, alg; cache_kwargs...).u
    @test eltype(u) <: Complex
    @test A2 * u ≈ b2
end

Random.seed!(10)
A = sprand(n, n, 0.8);
A2 = 2.0 .* A;
b1 = rand(n);
b2 = rand(n);
prob = LinearProblem(copy(A), copy(b1))

linsolve = init(prob, UMFPACKFactorization())
sol11 = solve!(linsolve)
linsolve.b = copy(b2)
sol12 = solve!(linsolve)
linsolve.A = copy(A2)
sol13 = solve!(linsolve)

for alg in algs
    local linsolve = init(prob, alg)
    sol31 = solve!(linsolve)
    linsolve.b = copy(b2)
    sol32 = solve!(linsolve)
    linsolve.A = copy(A2)
    sol33 = solve!(linsolve)
    @test sol11.u ≈ sol31.u
    @test sol12.u ≈ sol32.u
    @test sol13.u ≈ sol33.u
    adjoint_rhs = rand(n)
    adjoint_solution = LinearSolve._adjoint_factorization_solve(
        alg, linsolve.cacheval, linsolve.A, adjoint_rhs
    )
    @test adjoint(A2) * adjoint_solution ≈ adjoint_rhs

    complex_A = complex.(A2, sprand(n, n, 0.1))
    complex_b = rand(ComplexF64, n)
    complex_cache = init(LinearProblem(complex_A, complex_b), alg)
    solve!(complex_cache)
    complex_adjoint_rhs = rand(ComplexF64, n)
    complex_adjoint_solution = LinearSolve._adjoint_factorization_solve(
        alg, complex_cache.cacheval, complex_cache.A, complex_adjoint_rhs
    )
    @test adjoint(complex_A) * complex_adjoint_solution ≈ complex_adjoint_rhs
end

# Test for problem from #497
function makeA()
    n = 60
    colptr = [
        1, 4, 7, 11, 15, 17, 22, 26, 30, 34, 38, 40, 46, 50, 54, 58,
        62, 64, 70, 74, 78, 82, 86, 88, 94, 98, 102, 106, 110, 112,
        118, 122, 126, 130, 134, 136, 142, 146, 150, 154, 158, 160,
        166, 170, 174, 178, 182, 184, 190, 194, 198, 202, 206, 208,
        214, 218, 222, 224, 226, 228, 232,
    ]
    rowval = [
        1, 3, 4, 1, 2, 4, 2, 4, 9, 10, 3, 5, 11, 12, 1, 3, 2, 4, 6,
        11, 12, 2, 7, 9, 10, 2, 7, 8, 10, 8, 10, 15, 16, 9, 11, 17,
        18, 7, 9, 2, 8, 10, 12, 17, 18, 8, 13, 15, 16, 8, 13, 14, 16,
        14, 16, 21, 22, 15, 17, 23, 24, 13, 15, 8, 14, 16, 18, 23, 24,
        14, 19, 21, 22, 14, 19, 20, 22, 20, 22, 27, 28, 21, 23, 29, 30,
        19, 21, 14, 20, 22, 24, 29, 30, 20, 25, 27, 28, 20, 25, 26, 28,
        26, 28, 33, 34, 27, 29, 35, 36, 25, 27, 20, 26, 28, 30, 35, 36,
        26, 31, 33, 34, 26, 31, 32, 34, 32, 34, 39, 40, 33, 35, 41, 42,
        31, 33, 26, 32, 34, 36, 41, 42, 32, 37, 39, 40, 32, 37, 38, 40,
        38, 40, 45, 46, 39, 41, 47, 48, 37, 39, 32, 38, 40, 42, 47, 48,
        38, 43, 45, 46, 38, 43, 44, 46, 44, 46, 51, 52, 45, 47, 53, 54,
        43, 45, 38, 44, 46, 48, 53, 54, 44, 49, 51, 52, 44, 49, 50, 52,
        50, 52, 57, 58, 51, 53, 59, 60, 49, 51, 44, 50, 52, 54, 59, 60,
        50, 55, 57, 58, 50, 55, 56, 58, 56, 58, 57, 59, 55, 57, 50, 56,
        58, 60,
    ]
    nzval = [
        -0.64, 1.0, -1.0, 0.8606811145510832, -13.792569659442691, 1.0,
        0.03475000000000006, 1.0, -0.03510101010101016, -0.975,
        -1.0806825309567203, 1.0, -0.95, -0.025, 2.370597639417811,
        -2.3705976394178108, -11.083604432603583, -0.2770901108150896,
        1.0, -0.025, -0.95, -0.3564, -0.64, 1.0, -1.0, 13.792569659442691,
        0.8606811145510832, -13.792569659442691, 1.0, 0.03475000000000006,
        1.0, -0.03510101010101016, -0.975, -1.0806825309567203, 1.0, -0.95,
        -0.025, 2.370597639417811, -2.3705976394178108, 10.698449178570607,
        -11.083604432603583, -0.2770901108150896, 1.0, -0.025, -0.95, -0.3564,
        -0.64, 1.0, -1.0, 13.792569659442691, 0.8606811145510832,
        -13.792569659442691, 1.0, 0.03475000000000006, 1.0,
        -0.03510101010101016, -0.975, -1.0806825309567203, 1.0, -0.95,
        -0.025, 2.370597639417811, -2.3705976394178108, 10.698449178570607,
        -11.083604432603583, -0.2770901108150896, 1.0, -0.025, -0.95, -0.3564,
        -0.64, 1.0, -1.0, 13.792569659442691, 0.8606811145510832,
        -13.792569659442691, 1.0, 0.03475000000000006, 1.0, -0.03510101010101016,
        -0.975, -1.0806825309567203, 1.0, -0.95, -0.025, 2.370597639417811,
        -2.3705976394178108, 10.698449178570607, -11.083604432603583,
        -0.2770901108150896, 1.0, -0.025, -0.95, -0.3564, -0.64, 1.0,
        -1.0, 13.792569659442691, 0.8606811145510832, -13.792569659442691,
        1.0, 0.03475000000000006, 1.0, -0.03510101010101016, -0.975,
        -1.0806825309567203, 1.0, -0.95, -0.025, 2.370597639417811,
        -2.3705976394178108, 10.698449178570607, -11.083604432603583,
        -0.2770901108150896, 1.0, -0.025, -0.95, -0.3564, -0.64, 1.0,
        -1.0, 13.792569659442691, 0.8606811145510832, -13.792569659442691,
        1.0, 0.03475000000000006, 1.0, -0.03510101010101016, -0.975,
        -1.0806825309567203, 1.0, -0.95, -0.025, 2.370597639417811,
        -2.3705976394178108, 10.698449178570607, -11.083604432603583,
        -0.2770901108150896, 1.0, -0.025, -0.95, -0.3564, -0.64, 1.0,
        -1.0, 13.792569659442691, 0.8606811145510832, -13.792569659442691,
        1.0, 0.03475000000000006, 1.0, -0.03510101010101016, -0.975,
        -1.0806825309567203, 1.0, -0.95, -0.025, 2.370597639417811,
        -2.3705976394178108, 10.698449178570607, -11.083604432603583,
        -0.2770901108150896, 1.0, -0.025, -0.95, -0.3564, -0.64, 1.0,
        -1.0, 13.792569659442691, 0.8606811145510832, -13.792569659442691,
        1.0, 0.03475000000000006, 1.0, -0.03510101010101016, -0.975,
        -1.0806825309567203, 1.0, -0.95, -0.025, 2.370597639417811,
        -2.3705976394178108, 10.698449178570607, -11.083604432603583,
        -0.2770901108150896, 1.0, -0.025, -0.95, -0.3564, -0.64, 1.0,
        -1.0, 13.792569659442691, 0.8606811145510832, -13.792569659442691,
        1.0, 0.03475000000000006, 1.0, -0.03510101010101016, -0.975,
        -1.0806825309567203, 1.0, -0.95, -0.025, 2.370597639417811,
        -2.3705976394178108, 10.698449178570607, -11.083604432603583,
        -0.2770901108150896, 1.0, -0.025, -0.95, -0.3564, -0.64, 1.0,
        -1.0, 13.792569659442691, 0.8606811145510832, -13.792569659442691,
        1.0, 0.03475000000000006, 1.0, -1.0806825309567203, 1.0,
        2.370597639417811, -2.3705976394178108, 10.698449178570607,
        -11.083604432603583, -0.2770901108150896, 1.0,
    ]
    A = SparseMatrixCSC(n, n, colptr, rowval, nzval)
    return (A)
end

for alg in algs
    local A = makeA()
    u0 = fill(0.1, size(A, 2))
    linprob = LinearProblem(A, A * u0)
    u = LinearSolve.solve(linprob, alg)
    @test norm(u - u0) < 5.0e-14
end

# Testing and demonstrating Pardiso.set_iparm! for MKLPardisoSolver
for solver in solvers
    iparm = [
        (1, 1),
        (2, 2),
        (3, 0),
        (4, 0),
        (5, 0),
        (6, 0),
        (7, 0),
        (8, 20),
        (9, 0),
        (10, 13),
        (11, 1),
        (12, 1),
        (13, 1),
        (14, 0),
        (15, 0),
        (16, 0),
        (17, 0),
        (18, -1),
        (19, -1),
        (20, 0),
        (21, 0),
        (22, 0),
        (23, 0),
        (24, 10),
        (25, 0),
        (26, 0),
        (27, 1),
        (28, 0),
        (29, 0),
        (30, 0),
        (31, 0),
        (32, 0),
        (33, 0),
        (34, 0),
        (35, 0),
        (36, 0),
        (37, 0),
        (38, 0),
        (39, 0),
        (40, 0),
        (41, 0),
        (42, 0),
        (43, 0),
        (44, 0),
        (45, 0),
        (46, 0),
        (47, 0),
        (48, 0),
        (49, 0),
        (50, 0),
        (51, 0),
        (52, 0),
        (53, 0),
        (54, 0),
        (55, 0),
        (56, 0),
        (57, 0),
        (58, 0),
        (59, 0),
        (60, 0),
        (61, 0),
        (62, 0),
        (63, 0),
        (64, 0),
    ]

    for i in iparm
        Pardiso.set_iparm!(solver, i...)
    end

    for i in Base.OneTo(length(iparm))
        @test Pardiso.get_iparm(solver, i) == iparm[i][2]
    end
end

@testset "AbstractSparseMatrixCSC" begin
    struct MySparseMatrixCSC2{Tv, Ti} <: SparseArrays.AbstractSparseMatrixCSC{Tv, Ti}
        csc::SparseMatrixCSC{Tv, Ti}
    end

    Base.size(m::MySparseMatrixCSC2) = size(m.csc)
    SparseArrays.getcolptr(m::MySparseMatrixCSC2) = SparseArrays.getcolptr(m.csc)
    SparseArrays.rowvals(m::MySparseMatrixCSC2) = SparseArrays.rowvals(m.csc)
    SparseArrays.nonzeros(m::MySparseMatrixCSC2) = SparseArrays.nonzeros(m.csc)

    for alg in algs
        N = 100
        u0 = ones(N)
        A0 = spdiagm(1 => -ones(N - 1), 0 => fill(10.0, N), -1 => -ones(N - 1))
        b0 = A0 * u0
        B0 = MySparseMatrixCSC2(A0)
        A1 = spdiagm(1 => -ones(N - 1), 0 => fill(100.0, N), -1 => -ones(N - 1))
        b1 = A1 * u0
        B1 = MySparseMatrixCSC2(A1)

        pr = LinearProblem(B0, b0)
        # test default algorithn
        u = solve(pr, alg)
        @test norm(u - u0, Inf) < 1.0e-13

        # test factorization with reinit!
        pr = LinearProblem(B0, b0)
        cache = init(pr, alg)
        u = solve!(cache)
        @test norm(u - u0, Inf) < 1.0e-13
        reinit!(cache; A = B1, b = b1)
        u = solve!(cache)
        @test norm(u - u0, Inf) < 1.0e-13
    end
end
# SuiteSparse HB/dwt_59 — symmetric; MKL REAL_NONSYM returns wrong solution
const dwt59_n = 59
const dwt59_colptr = [1, 5, 10, 13, 16, 21, 25, 30, 35, 40, 45, 49, 53, 58, 63, 67, 72, 77, 82, 87, 91, 96, 101, 106, 112, 117, 121, 126, 131, 136, 141, 146, 151, 155, 159, 164, 170, 175, 179, 184, 189, 194, 199, 201, 205, 210, 212, 217, 222, 228, 233, 238, 243, 248, 250, 254, 256, 258, 263, 268]
const dwt59_rowval = [1, 2, 7, 9, 1, 2, 3, 7, 10, 2, 3, 11, 4, 5, 12, 4, 5, 6, 8, 13, 5, 6, 8, 14, 1, 2, 7, 9, 10, 5, 6, 8, 13, 14, 1, 7, 9, 10, 27, 2, 7, 9, 10, 11, 3, 10, 11, 15, 4, 12, 13, 20, 5, 8, 12, 13, 14, 6, 8, 13, 14, 30, 11, 15, 16, 34, 15, 16, 17, 21, 35, 16, 17, 18, 21, 23, 17, 18, 19, 22, 24, 18, 19, 20, 22, 25, 12, 19, 20, 26, 16, 17, 21, 23, 35, 18, 19, 22, 24, 25, 17, 21, 23, 24, 35, 18, 22, 23, 24, 25, 36, 19, 22, 24, 25, 26, 20, 25, 26, 29, 9, 27, 28, 31, 39, 27, 28, 31, 33, 34, 26, 29, 30, 32, 38, 14, 29, 30, 32, 42, 27, 28, 31, 39, 40, 29, 30, 32, 41, 42, 28, 33, 40, 58, 15, 28, 34, 35, 16, 21, 23, 34, 35, 24, 36, 37, 50, 51, 59, 36, 37, 38, 51, 52, 29, 37, 38, 41, 27, 31, 39, 40, 44, 31, 33, 39, 40, 45, 32, 38, 41, 42, 53, 30, 32, 41, 42, 55, 43, 44, 39, 43, 44, 45, 40, 44, 45, 46, 47, 45, 46, 45, 47, 48, 49, 58, 47, 48, 49, 58, 59, 47, 48, 49, 50, 57, 59, 36, 49, 50, 51, 52, 36, 37, 50, 51, 52, 37, 50, 51, 52, 53, 41, 52, 53, 54, 55, 53, 54, 42, 53, 55, 56, 55, 56, 49, 57, 33, 47, 48, 58, 59, 36, 48, 49, 58, 59]
const dwt59_nzval = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# Regression: symmetric SuiteSparse graphs (HB/dwt_59). MKL REAL_NONSYM returns a
# catastrophically wrong solution here; the wrapper must select a symmetric Pardiso
# matrix type and pass Pardiso.get_matrix (triangular) storage.
# Reproducer context: SciMLBenchmarks MatrixDepot LinearSolve benchmark.
@testset "symmetric SuiteSparse matrix type (HB/dwt_59)" begin
    A_sym = SparseMatrixCSC(dwt59_n, dwt59_n, dwt59_colptr, dwt59_rowval, dwt59_nzval)
    @test issymmetric(A_sym)
    b_sym = rand(MersenneTwister(123), dwt59_n)
    refres = norm(A_sym * (Matrix(A_sym) \ b_sym) - b_sym) / norm(b_sym)

    if Pardiso.mkl_is_available()
        # Old default path (forced nonsymmetric) is inaccurate on this matrix.
        bad = solve(LinearProblem(copy(A_sym), copy(b_sym)),
            MKLPardisoFactorize(matrix_type = Pardiso.REAL_NONSYM))
        @test norm(A_sym * bad.u - b_sym) / norm(b_sym) > max(1e-6, 10 * refres)

        cache = init(LinearProblem(copy(A_sym), copy(b_sym)), MKLPardisoFactorize())
        @test Pardiso.get_matrixtype(cache.cacheval) == Pardiso.REAL_SYM_INDEF
        sol = solve!(cache)
        @test norm(A_sym * sol.u - b_sym) / norm(b_sym) <= max(1e-10, 10 * refres)
    end

    if Pardiso.panua_is_available()
        sol = solve(LinearProblem(copy(A_sym), copy(b_sym)), PanuaPardisoFactorize())
        @test norm(A_sym * sol.u - b_sym) / norm(b_sym) <= max(1e-10, 10 * refres)
    end
end
