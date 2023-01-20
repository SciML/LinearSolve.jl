using HYPRE
using LinearSolve
using MPI
using SparseArrays
using Test

MPI.Init()
HYPRE.Init()

const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const comm_size = MPI.Comm_size(comm)

if comm_size != 2
    error("must run with 2 ranks")
end

if rank == 0
    ilower = 1
    iupper = 10
else
    ilower = 11
    iupper = 20
end
local_size = iupper - ilower + 1
local_sol = Vector{Float64}(undef, local_size)

# Create the matrix and vector
function getAb(scaling)
    A = HYPREMatrix(comm, ilower, iupper)
    b = HYPREVector(comm, ilower, iupper)
    assembler = HYPRE.start_assemble!(A, b)
    for idx in ilower:iupper
        a = fill(1.0, 1, 1)
        c = fill(scaling * idx, 1)
        HYPRE.assemble!(assembler, [idx], a, c)
    end
    HYPRE.finish_assemble!(assembler)
    return A, b
end

const TOL = LinearSolve.default_tol(HYPRE.LibHYPRE.HYPRE_Complex)

# Solve without initial guess (GMRES)
A, b = getAb(1.0)
alg = HYPREAlgorithm(HYPRE.GMRES)
prob = LinearProblem(A, b)
sol = solve(prob, alg)
@test sol.resid < TOL
@test sol.iters > 0
copy!(local_sol, sol.u)
@test local_sol ≈ ilower:iupper

# Solve with initial guess (PCG)
A, b = getAb(2.0)
alg = HYPREAlgorithm(HYPRE.PCG)
prob = LinearProblem(A, b; u0 = zero(b))
sol = solve(prob, alg)
@test sol.resid < TOL
@test sol.iters > 0
copy!(local_sol, sol.u)
@test local_sol ≈ 2 * (ilower:iupper)

# Solve with cache (BiCGSTAB)
A, b = getAb(3.0)
alg = HYPREAlgorithm(HYPRE.BiCGSTAB)
prob = LinearProblem(A, b)
cache = init(prob, alg)
sol = solve(cache)
@test sol.resid < TOL
@test sol.iters > 0
copy!(local_sol, sol.u)
@test local_sol ≈ 3 * (ilower:iupper)

# Solve after updated b
_, b = getAb(4.0)
cache = LinearSolve.set_b(sol.cache, b)
sol = solve(cache)
@test sol.resid < TOL
@test sol.iters > 0
copy!(local_sol, sol.u)
@test local_sol ≈ 4 * (ilower:iupper)
