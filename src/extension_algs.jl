# This file only include the algorithm struct to be exported by LinearSolve.jl. The main
# functionality is implemented as package extensions

struct HYPREAlgorithm <: SciMLLinearSolveAlgorithm
    solver::Any

    @static if VERSION >= v"1.9-"
        function HYPREAlgorithm(solver)
            ext = Base.get_extension(@__MODULE__, :LinearSolveHYPREExt)
            if ext === nothing
                error("HYPREAlgorithm requires that HYPRE is loaded, i.e. `using HYPRE`")
            else
                return new{}(solver)
            end
        end
    end
end

struct CudaOffloadFactorization <: LinearSolve.AbstractFactorization 
    @static if VERSION >= v"1.9-"
        function CudaOffloadFactorization()
            ext = Base.get_extension(@__MODULE__, :LinearSolveCUDAExt)
            if ext === nothing
                error("CudaOffloadFactorization requires that CUDA is loaded, i.e. `using CUDA`")
            else
                return new{}()
            end
        end
    end
end

MKLPardisoFactorize(; kwargs...) = PardisoJL(; solver_type = 0, kwargs...)
MKLPardisoIterate(; kwargs...) = PardisoJL(; solver_type = 1, kwargs...)

@static if VERSION >= v"1.9-"
    struct PardisoJL{T1,T2} <: LinearSolve.SciMLLinearSolveAlgorithm
        nprocs::Union{Int, Nothing}
        solver_type::T1
        matrix_type::T2
        iparm::Union{Vector{Tuple{Int, Int}}, Nothing}
        dparm::Union{Vector{Tuple{Int, Int}}, Nothing}

        function PardisoJL(;nprocs::Union{Int, Nothing} = nothing,
            solver_type = nothing,
            matrix_type = nothing,
            iparm::Union{Vector{Tuple{Int, Int}}, Nothing} = nothing,
            dparm::Union{Vector{Tuple{Int, Int}}, Nothing} = nothing)

            ext = Base.get_extension(@__MODULE__, :LinearSolvePardisoExt)
            if ext === nothing
                error("PardisoJL requires that Pardiso is loaded, i.e. `using Pardiso`")
            else
                T1 = typeof(solver_type)
                T2 = typeof(matrix_type)
                @assert T1 <: Union{Int, Nothing, ext.Pardiso.Solver}
                @assert T2 <: Union{Int, Nothing, ext.Pardiso.MatrixType}
                return new{T1, T2}(nprocs, solver_type, matrix_type, iparm, dparm)
            end
        end
    end
else
    Base.@kwdef struct PardisoJL <: LinearSolve.SciMLLinearSolveAlgorithm
        nprocs::Union{Int, Nothing} = nothing
        solver_type::Any = nothing
        matrix_type::Any = nothing
        iparm::Union{Vector{Tuple{Int, Int}}, Nothing} = nothing
        dparm::Union{Vector{Tuple{Int, Int}}, Nothing} = nothing
    end    
end