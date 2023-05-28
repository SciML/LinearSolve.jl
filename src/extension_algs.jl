# This file only include the algorithm struct to be exported by LinearSolve.jl. The main
# functionality is implemented as a package extension in ext/LinearSolveHYPRE.jl.

struct HYPREAlgorithm <: SciMLLinearSolveAlgorithm
    solver::Any
end

struct CudaOffloadFactorization <: LinearSolve.AbstractFactorization end

MKLPardisoFactorize(; kwargs...) = PardisoJL(; solver_type = 0, kwargs...)
MKLPardisoIterate(; kwargs...) = PardisoJL(; solver_type = 1, kwargs...)
