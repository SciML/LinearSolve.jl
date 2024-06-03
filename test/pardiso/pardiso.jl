using LinearSolve, SparseArrays, Random
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

algs=[PardisoJL()]

if Pardiso.mkl_is_available()
    algs=vcat(algs,[MKLPardisoFactorize(), MKLPardisoIterate()])
end
    
if Pardiso.panua_is_available()
    algs=vcat(algs,[PanuaPardisoFactorize(), PanuaPardisoIterate()])
end    


for alg in algs
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

prob = LinearProblem(copy(A), copy(b1))
linsolve = init(prob, UMFPACKFactorization())
sol11 = solve(linsolve)
linsolve = LinearSolve.set_b(sol11.cache, copy(b2))
sol12 = solve(linsolve)
linsolve = LinearSolve.set_A(sol12.cache, copy(A2))
sol13 = solve(linsolve)

linsolve = init(prob, MKLPardisoFactorize())
sol31 = solve(linsolve)
linsolve = LinearSolve.set_b(sol31.cache, copy(b2))
sol32 = solve(linsolve)
linsolve = LinearSolve.set_A(sol32.cache, copy(A2))
sol33 = solve(linsolve)

@test sol11.u ≈ sol31.u
@test sol12.u ≈ sol32.u
@test sol13.u ≈ sol33.u

# Testing and demonstrating Pardiso.set_iparm! for MKLPardisoSolver
solver = Pardiso.MKLPardisoSolver()
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
