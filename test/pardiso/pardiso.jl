using LinearSolve, SparseArrays, Random, LinearAlgebra
import Pardiso

A1 = sparse([1.0 0 -2 3
             0 5 1 2
             -2 1 4 -7
             3 2 -7 5])
b1 = rand(4)
prob1 = LinearProblem(A1, b1)

lambda = 3
n = 4
e = ones(n)
e2 = ones(n - 1)
A2 = spdiagm(-1 => im * e2, 0 => lambda * e, 1 => -im * e2)
b2 = rand(n) + im * zeros(n)
cache_kwargs = (; abstol = 1e-8, reltol = 1e-8, maxiter = 30)

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
linsolve = LinearSolve.set_b(sol11.cache, copy(b2))
sol12 = solve!(linsolve)
linsolve = LinearSolve.set_A(sol12.cache, copy(A2))
sol13 = solve!(linsolve)

for alg in algs
    linsolve = init(prob, alg)
    sol31 = solve!(linsolve)
    linsolve = LinearSolve.set_b(sol31.cache, copy(b2))
    sol32 = solve!(linsolve)
    linsolve = LinearSolve.set_A(sol32.cache, copy(A2))
    sol33 = solve!(linsolve)
    @test sol11.u ≈ sol31.u
    @test sol12.u ≈ sol32.u
    @test sol13.u ≈ sol33.u
end

# Test for problem from #497
function makeA()
    n = 60
    colptr = [1, 4, 7, 11, 15, 17, 22, 26, 30, 34, 38, 40, 46, 50, 54, 58,
        62, 64, 70, 74, 78, 82, 86, 88, 94, 98, 102, 106, 110, 112,
        118, 122, 126, 130, 134, 136, 142, 146, 150, 154, 158, 160,
        166, 170, 174, 178, 182, 184, 190, 194, 198, 202, 206, 208,
        214, 218, 222, 224, 226, 228, 232]
    rowval = [1, 3, 4, 1, 2, 4, 2, 4, 9, 10, 3, 5, 11, 12, 1, 3, 2, 4, 6,
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
        58, 60]
    nzval = [-0.64, 1.0, -1.0, 0.8606811145510832, -13.792569659442691, 1.0,
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
        -11.083604432603583, -0.2770901108150896, 1.0]
    A = SparseMatrixCSC(n, n, colptr, rowval, nzval)
    return (A)
end

for alg in algs
    A = makeA()
    u0 = fill(0.1, size(A, 2))
    linprob = LinearProblem(A, A * u0)
    u = LinearSolve.solve(linprob, alg)
    @test norm(u - u0) < 1.0e-14
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
        (64, 0)
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
