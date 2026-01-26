module LinearSolvePETScExt

using LinearAlgebra
using MPI
using PETSc
using PETSc: petsclibs, PetscScalar, PetscInt
using SparseArrays: SparseMatrixCSC, sparse
using LinearSolve: PETScAlgorithm, LinearCache, LinearProblem, LinearSolve,
    OperatorAssumptions, default_tol, init_cacheval, __issquare,
    __conditioning, LinearSolveAdjoint, LinearVerbosity
using SciMLLogging: SciMLLogging, verbosity_to_int, @SciMLMessage
using SciMLBase: LinearProblem, LinearAliasSpecifier, SciMLBase
using Setfield: @set!

mutable struct PETScCache
    ksp::Union{PETSc.AbstractKSP, Nothing}
    petsclib::Any
    A::Union{PETSc.AbstractMat, Nothing}
    b::Union{PETSc.AbstractVec, Nothing}
    u::Union{PETSc.AbstractVec, Nothing}
    initialized::Bool
end

function LinearSolve.init_cacheval(
        alg::PETScAlgorithm, A, b, u, Pl, Pr, maxiters::Int,
        abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return PETScCache(nothing, nothing, nothing, nothing, nothing, false)
end

# Helper function to get or initialize the PETSc library
function get_petsclib(T::Type = Float64)
    # Find a petsclib that matches our scalar type
    for lib in PETSc.petsclibs
        if PETSc.scalartype(lib) === T
            return lib
        end
    end
    # Fallback to first available
    return first(PETSc.petsclibs)
end

# Convert Julia matrix to PETSc matrix
function to_petsc_mat(petsclib, A::SparseMatrixCSC{T}) where {T}
    m, n = size(A)
    mat = PETSc.MatSeqAIJ(petsclib, A)
    return mat
end

function to_petsc_mat(petsclib, A::AbstractMatrix{T}) where {T}
    # Convert dense matrix to sparse first
    return to_petsc_mat(petsclib, sparse(A))
end

# Convert Julia vector to PETSc vector
function to_petsc_vec(petsclib, v::AbstractVector{T}) where {T}
    return PETSc.VecSeq(petsclib, v)
end

# Symbol to PETSc KSP type string
function ksp_type_string(solver_type::Symbol)
    ksp_types = Dict(
        :gmres => "gmres",
        :cg => "cg",
        :bicg => "bicg",
        :bcgs => "bcgs",
        :bcgsl => "bcgsl",
        :cgs => "cgs",
        :tfqmr => "tfqmr",
        :cr => "cr",
        :gcr => "gcr",
        :preonly => "preonly",
        :richardson => "richardson",
        :chebyshev => "chebyshev",
        :minres => "minres",
        :symmlq => "symmlq",
        :lgmres => "lgmres",
        :fgmres => "fgmres",
    )
    return get(ksp_types, solver_type, string(solver_type))
end

# Symbol to PETSc PC type string
function pc_type_string(pc_type::Symbol)
    pc_types = Dict(
        :none => "none",
        :jacobi => "jacobi",
        :sor => "sor",
        :lu => "lu",
        :ilu => "ilu",
        :icc => "icc",
        :cholesky => "cholesky",
        :gamg => "gamg",
        :hypre => "hypre",
        :bjacobi => "bjacobi",
        :asm => "asm",
        :mg => "mg",
    )
    return get(pc_types, pc_type, string(pc_type))
end

function SciMLBase.solve!(cache::LinearCache, alg::PETScAlgorithm, args...; kwargs...)
    pcache = cache.cacheval

    # Get element type from the problem
    T = eltype(cache.A)

    # Initialize PETSc if needed
    if !pcache.initialized
        petsclib = get_petsclib(T)
        if !PETSc.initialized(petsclib)
            PETSc.initialize(petsclib)
        end
        pcache.petsclib = petsclib
        pcache.initialized = true
    end

    petsclib = pcache.petsclib

    # Convert matrix if needed
    if pcache.A === nothing || cache.isfresh
        pcache.A = to_petsc_mat(petsclib, cache.A)
    end

    # Convert vectors
    pcache.b = to_petsc_vec(petsclib, cache.b)
    pcache.u = to_petsc_vec(petsclib, copy(cache.u))

    # Create KSP solver if needed
    if pcache.ksp === nothing
        ksp = PETSc.KSP(petsclib, pcache.A; ksp_rtol = cache.reltol, ksp_atol = cache.abstol)

        # Set KSP type
        ksp_type = ksp_type_string(alg.solver_type)
        PETSc.settype!(ksp, ksp_type)

        # Set PC type
        if alg.pc_type !== :none
            pc = PETSc.PC(ksp)
            pc_type = pc_type_string(alg.pc_type)
            PETSc.settype!(pc, pc_type)
        end

        # Set max iterations
        PETSc.settolerances!(ksp; maxits = cache.maxiters)

        PETSc.setfromoptions!(ksp)
        pcache.ksp = ksp
    else
        # Update operators if matrix changed
        if cache.isfresh
            PETSc.setoperators!(pcache.ksp, pcache.A, pcache.A)
        end
    end

    cache.isfresh = false

    # Solve
    PETSc.solve!(pcache.u, pcache.ksp, pcache.b)

    # Copy solution back to Julia vector
    copy!(cache.u, pcache.u)

    # Get convergence info
    iters = PETSc.getiterationnumber(pcache.ksp)
    reason = PETSc.getconvergedreason(pcache.ksp)
    resid = PETSc.getresidualnorm(pcache.ksp)

    # Determine return code
    retcode = if reason > 0
        SciMLBase.ReturnCode.Success
    elseif reason == 0
        SciMLBase.ReturnCode.Default
    else
        SciMLBase.ReturnCode.Failure
    end

    stats = nothing
    return SciMLBase.build_linear_solution(alg, cache.u, resid, cache; retcode, iters, stats)
end

end # module LinearSolvePETScExt
