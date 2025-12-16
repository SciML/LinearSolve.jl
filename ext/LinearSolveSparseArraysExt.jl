module LinearSolveSparseArraysExt

using LinearSolve: LinearSolve, BLASELTYPES, pattern_changed, ArrayInterface,
                   @get_cacheval, CHOLMODFactorization, GenericFactorization,
                   GenericLUFactorization,
                   KLUFactorization, LUFactorization, NormalCholeskyFactorization,
                   OperatorAssumptions, LinearVerbosity,
                   QRFactorization, RFLUFactorization, UMFPACKFactorization, solve
using SciMLOperators: AbstractSciMLOperator, has_concretization
using ArrayInterface: ArrayInterface
using LinearAlgebra: LinearAlgebra, I, Hermitian, Symmetric, cholesky, ldiv!, lu, lu!, QR
using SparseArrays: SparseArrays, AbstractSparseArray, AbstractSparseMatrixCSC,
                    SparseMatrixCSC,
                    nonzeros, rowvals, getcolptr, sparse, sprand
using SciMLLogging: @SciMLMessage

@static if Base.USE_GPL_LIBS
    using SparseArrays.UMFPACK: UMFPACK_OK
end
using Base: /, \, convert
using SciMLBase: SciMLBase, LinearProblem, ReturnCode
import StaticArraysCore: SVector

# Can't `using KLU` because cannot have a dependency in there without
# requiring the user does `using KLU`
# But there's no reason to require it because SparseArrays will already
# load SuiteSparse and thus all of the underlying KLU code
include("../src/KLU/klu.jl")

LinearSolve.issparsematrixcsc(A::AbstractSparseMatrixCSC) = true
LinearSolve.issparsematrix(A::AbstractSparseArray) = true
function LinearSolve.make_SparseMatrixCSC(A::AbstractSparseArray)
    SparseMatrixCSC(size(A)..., getcolptr(A), rowvals(A), nonzeros(A))
end
function LinearSolve.makeempty_SparseMatrixCSC(A::AbstractSparseArray)
    SparseMatrixCSC(0, 0, [1], Int[], eltype(A)[])
end

function LinearSolve.init_cacheval(alg::RFLUFactorization,
        A::Union{AbstractSparseArray, LinearSolve.SciMLOperators.AbstractSciMLOperator}, b, u, Pl, Pr,
        maxiters::Int,
        abstol, reltol, verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions)
    nothing, nothing
end

function LinearSolve.handle_sparsematrixcsc_lu(A::AbstractSparseMatrixCSC)
    @static if Base.USE_GPL_LIBS
        lu(SparseMatrixCSC(size(A)..., getcolptr(A), rowvals(A), nonzeros(A)),
            check = false)
    else
        error("Sparse LU factorization requires GPL libraries (UMFPACK). Use `using Sparspak` for a non-GPL alternative or rebuild Julia with USE_GPL_LIBS=1")
    end
end

@static if Base.USE_GPL_LIBS
    function LinearSolve.defaultalg(
            A::Symmetric{<:BLASELTYPES, <:SparseMatrixCSC}, b, ::OperatorAssumptions{Bool})
        LinearSolve.DefaultLinearSolver(LinearSolve.DefaultAlgorithmChoice.CHOLMODFactorization)
    end
else
    function LinearSolve.defaultalg(
            A::Symmetric{<:BLASELTYPES, <:SparseMatrixCSC}, b, ::OperatorAssumptions{Bool})
        LinearSolve.DefaultLinearSolver(LinearSolve.DefaultAlgorithmChoice.CholeskyFactorization)
    end
end # @static if Base.USE_GPL_LIBS

function LinearSolve.defaultalg(A::AbstractSparseMatrixCSC{Tv, Ti}, b,
        assump::OperatorAssumptions{Bool}) where {Tv, Ti}
    ext = Base.get_extension(LinearSolve, :LinearSolveSparspakExt)
    if assump.issq && ext !== nothing
        LinearSolve.DefaultLinearSolver(LinearSolve.DefaultAlgorithmChoice.SparspakFactorization)
    elseif !assump.issq
        error("Generic number sparse factorization for non-square is not currently handled")
    elseif ext === nothing
        error("SparspakFactorization required for general sparse matrix types and with general Julia number types. Do `using Sparspak` in order to enable this functionality")
    else
        error("Unreachable reached. Please report this error with a reproducer.")
    end
end

function LinearSolve.init_cacheval(alg::GenericFactorization,
        A::Union{Hermitian{T, <:SparseMatrixCSC},
            Symmetric{T, <:SparseMatrixCSC}}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions) where {T}
    newA = copy(convert(AbstractMatrix, A))
    LinearSolve.do_factorization(alg, newA, b, u)
end

@static if Base.USE_GPL_LIBS
    const PREALLOCATED_UMFPACK = SparseArrays.UMFPACK.UmfpackLU(SparseMatrixCSC(0, 0, [1],
        Int[], Float64[]))
end # @static if Base.USE_GPL_LIBS

function LinearSolve.init_cacheval(
        alg::LUFactorization, A::AbstractSparseArray{<:Number, <:Integer}, b, u,
        Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions)
    nothing
end

function LinearSolve.init_cacheval(
        alg::GenericLUFactorization, A::AbstractSparseArray{<:Number, <:Integer}, b, u,
        Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions)
    nothing
end

function LinearSolve.init_cacheval(
        alg::UMFPACKFactorization, A::AbstractArray, b, u,
        Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions)
    nothing
end

@static if Base.USE_GPL_LIBS
    function LinearSolve.init_cacheval(
            alg::LUFactorization, A::AbstractSparseArray{Float64, Int64}, b, u,
            Pl, Pr,
            maxiters::Int, abstol, reltol,
            verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions)
        PREALLOCATED_UMFPACK
    end
    function LinearSolve.init_cacheval(
            alg::LUFactorization, A::AbstractSparseArray{T, Int64}, b, u,
            Pl, Pr,
            maxiters::Int, abstol, reltol,
            verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions) where {T <:
                                                                                            BLASELTYPES}
        if LinearSolve.is_cusparse(A)
            LinearSolve.cudss_loaded(A) ? ArrayInterface.lu_instance(A) : nothing
        else
            SparseArrays.UMFPACK.UmfpackLU(SparseMatrixCSC{T, Int64}(
                zero(Int64), zero(Int64), [Int64(1)], Int64[], T[]))
        end
    end
    function LinearSolve.init_cacheval(
            alg::LUFactorization, A::AbstractSparseArray{T, Int32}, b, u,
            Pl, Pr,
            maxiters::Int, abstol, reltol,
            verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions) where {T <:
                                                                                            BLASELTYPES}
        if LinearSolve.is_cusparse(A)
            LinearSolve.cudss_loaded(A) ? ArrayInterface.lu_instance(A) : nothing
        else
            SparseArrays.UMFPACK.UmfpackLU(SparseMatrixCSC{T, Int32}(
                zero(Int32), zero(Int32), [Int32(1)], Int32[], T[]))
        end
    end
end # @static if Base.USE_GPL_LIBS

function LinearSolve.init_cacheval(
        alg::LUFactorization, A::LinearSolve.GPUArraysCore.AnyGPUArray, b, u,
        Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions)
    ArrayInterface.lu_instance(A)
end

function LinearSolve.init_cacheval(
        alg::UMFPACKFactorization, A::LinearSolve.GPUArraysCore.AnyGPUArray, b, u,
        Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions)
    nothing
end

@static if Base.USE_GPL_LIBS
    function LinearSolve.init_cacheval(
            alg::UMFPACKFactorization, A::AbstractSparseArray{Float64, Int}, b, u, Pl, Pr,
            maxiters::Int, abstol,
            reltol,
            verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions)
        PREALLOCATED_UMFPACK
    end

    function LinearSolve.init_cacheval(
            alg::UMFPACKFactorization, A::AbstractSparseArray{T, Int64}, b, u,
            Pl, Pr,
            maxiters::Int, abstol, reltol,
            verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions) where {T <:
                                                                                            BLASELTYPES}
        SparseArrays.UMFPACK.UmfpackLU(SparseMatrixCSC{T, Int64}(
            zero(Int64), zero(Int64), [Int64(1)], Int64[], T[]))
    end

    function LinearSolve.init_cacheval(
            alg::UMFPACKFactorization, A::AbstractSparseArray{T, Int32}, b, u,
            Pl, Pr,
            maxiters::Int, abstol, reltol,
            verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions) where {T <:
                                                                                            BLASELTYPES}
        SparseArrays.UMFPACK.UmfpackLU(SparseMatrixCSC{T, Int32}(
            zero(Int32), zero(Int32), [Int32(1)], Int32[], T[]))
    end

    function SciMLBase.solve!(
            cache::LinearSolve.LinearCache, alg::UMFPACKFactorization; kwargs...)
        A = cache.A
        A = convert(AbstractMatrix, A)
        if cache.isfresh
            cacheval = LinearSolve.@get_cacheval(cache, :UMFPACKFactorization)
            if alg.reuse_symbolic
                # Caches the symbolic factorization: https://github.com/JuliaLang/julia/pull/33738
                if length(cacheval.nzval) != length(nonzeros(A)) ||
                   alg.check_pattern && pattern_changed(cacheval, A)
                    fact = lu(
                        SparseMatrixCSC(size(A)..., getcolptr(A), rowvals(A),
                            nonzeros(A)),
                        check = false)
                else
                    fact = lu!(cacheval,
                        SparseMatrixCSC(size(A)..., getcolptr(A), rowvals(A),
                            nonzeros(A)), check = false)
                end
            else
                fact = lu(
                    SparseMatrixCSC(size(A)..., getcolptr(A), rowvals(A), nonzeros(A)),
                    check = false)
            end
            cache.cacheval = fact
            cache.isfresh = false
        end

        F = LinearSolve.@get_cacheval(cache, :UMFPACKFactorization)
        if F.status == UMFPACK_OK
            y = ldiv!(cache.u, F, cache.b)
            SciMLBase.build_linear_solution(
                alg, y, nothing, cache; retcode = ReturnCode.Success)
        else
            @SciMLMessage("Solver failed", cache.verbose, :solver_failure)
            SciMLBase.build_linear_solution(
                alg, cache.u, nothing, cache; retcode = ReturnCode.Infeasible)
        end
    end

else
    function SciMLBase.solve!(
            cache::LinearSolve.LinearCache, alg::UMFPACKFactorization; kwargs...)
        error("UMFPACKFactorization requires GPL libraries (UMFPACK). Rebuild Julia with USE_GPL_LIBS=1 or use an alternative algorithm like SparspakFactorization")
    end
end # @static if Base.USE_GPL_LIBS

function LinearSolve.init_cacheval(
        alg::KLUFactorization, A::AbstractArray, b, u, Pl,
        Pr,
        maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions)
    nothing
end

function LinearSolve.init_cacheval(
        alg::KLUFactorization, A::LinearSolve.GPUArraysCore.AnyGPUArray, b, u,
        Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions)
    nothing
end

const PREALLOCATED_KLU = KLU.KLUFactorization(SparseMatrixCSC(0, 0, [1], Int[],
    Float64[]))

function LinearSolve.init_cacheval(
        alg::KLUFactorization, A::AbstractSparseArray{Float64, Int64}, b, u, Pl, Pr,
        maxiters::Int, abstol,
        reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions)
    PREALLOCATED_KLU
end

function LinearSolve.init_cacheval(
        alg::KLUFactorization, A::AbstractSparseArray{Float64, Int32}, b, u, Pl, Pr,
        maxiters::Int, abstol,
        reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions)
    KLU.KLUFactorization(SparseMatrixCSC{Float64, Int32}(
        0, 0, [Int32(1)], Int32[], Float64[]))
end

# AbstractSciMLOperator handling for sparse factorizations
function LinearSolve.init_cacheval(
        alg::KLUFactorization, A::AbstractSciMLOperator, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions)
    if has_concretization(A)
        return LinearSolve.init_cacheval(alg, convert(AbstractMatrix, A), b, u, Pl, Pr,
            maxiters, abstol, reltol, verbose, assumptions)
    else
        nothing
    end
end

function LinearSolve.init_cacheval(
        alg::UMFPACKFactorization, A::AbstractSciMLOperator, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions)
    if has_concretization(A)
        return LinearSolve.init_cacheval(alg, convert(AbstractMatrix, A), b, u, Pl, Pr,
            maxiters, abstol, reltol, verbose, assumptions)
    else
        nothing
    end
end

function LinearSolve.init_cacheval(
        alg::CHOLMODFactorization, A::AbstractSciMLOperator, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions)
    if has_concretization(A)
        return LinearSolve.init_cacheval(alg, convert(AbstractMatrix, A), b, u, Pl, Pr,
            maxiters, abstol, reltol, verbose, assumptions)
    else
        nothing
    end
end

function LinearSolve.init_cacheval(
        alg::NormalCholeskyFactorization, A::AbstractSciMLOperator, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions)
    if has_concretization(A)
        return LinearSolve.init_cacheval(alg, convert(AbstractMatrix, A), b, u, Pl, Pr,
            maxiters, abstol, reltol, verbose, assumptions)
    else
        nothing
    end
end

function SciMLBase.solve!(cache::LinearSolve.LinearCache, alg::KLUFactorization; kwargs...)
    A = cache.A
    A = convert(AbstractMatrix, A)
    if cache.isfresh
        cacheval = LinearSolve.@get_cacheval(cache, :KLUFactorization)
        if alg.reuse_symbolic
            if length(cacheval.nzval) != length(nonzeros(A)) ||
               alg.check_pattern && pattern_changed(cacheval, A)
                fact = KLU.klu(
                    SparseMatrixCSC(size(A)..., getcolptr(A), rowvals(A),
                        nonzeros(A)),
                    check = false)
            else
                fact = KLU.klu!(cacheval, nonzeros(A), check = false)
            end
        else
            # New fact each time since the sparsity pattern can change
            # and thus it needs to reallocate
            fact = KLU.klu(SparseMatrixCSC(size(A)..., getcolptr(A), rowvals(A),
                nonzeros(A)))
        end
        cache.cacheval = fact
        cache.isfresh = false
    end
    F = LinearSolve.@get_cacheval(cache, :KLUFactorization)
    if F.common.status == KLU.KLU_OK
        y = ldiv!(cache.u, F, cache.b)
        SciMLBase.build_linear_solution(
            alg, y, nothing, cache; retcode = ReturnCode.Success)
    else
        @SciMLMessage("Solver failed", cache.verbose, :solver_failure)
        SciMLBase.build_linear_solution(
            alg, cache.u, nothing, cache; retcode = ReturnCode.Infeasible)
    end
end

function LinearSolve.init_cacheval(alg::CHOLMODFactorization,
        A::AbstractArray, b, u,
        Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions)
    nothing
end

@static if Base.USE_GPL_LIBS
    const PREALLOCATED_CHOLMOD = cholesky(sparse(reshape([1.0], 1, 1)))

    function LinearSolve.init_cacheval(alg::CHOLMODFactorization,
            A::Union{SparseMatrixCSC{T, Int}, Symmetric{T, SparseMatrixCSC{T, Int}}}, b, u,
            Pl, Pr,
            maxiters::Int, abstol, reltol,
            verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions) where {T <:
                                                                                            Float64}
        PREALLOCATED_CHOLMOD
    end

    function LinearSolve.init_cacheval(alg::CHOLMODFactorization,
            A::Union{SparseMatrixCSC{T, Int}, Symmetric{T, SparseMatrixCSC{T, Int}}}, b, u,
            Pl, Pr,
            maxiters::Int, abstol, reltol,
            verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions) where {T <:
                                                                                            BLASELTYPES}
        cholesky(sparse(reshape([one(T)], 1, 1)))
    end
end # @static if Base.USE_GPL_LIBS

function LinearSolve.init_cacheval(alg::NormalCholeskyFactorization,
        A::Union{AbstractSparseArray{T}, LinearSolve.GPUArraysCore.AnyGPUArray,
            Symmetric{T, <:AbstractSparseArray{T}}}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions) where {T <: BLASELTYPES}
    if LinearSolve.is_cusparse_csc(A)
        nothing
    elseif LinearSolve.is_cusparse_csr(A) && !LinearSolve.cudss_loaded(A)
        nothing
    elseif !assumptions.issq
        # Cholesky requires square matrices - return nothing for non-square to avoid errors
        # during DefaultLinearSolver cache initialization
        # See https://github.com/SciML/NonlinearSolve.jl/issues/746
        nothing
    else
        ArrayInterface.cholesky_instance(convert(AbstractMatrix, A))
    end
end

# Specialize QR for the non-square case
# Missing ldiv! definitions: https://github.com/JuliaSparse/SparseArrays.jl/issues/242
function LinearSolve._ldiv!(x::Vector,
        A::Union{QR, LinearAlgebra.QRCompactWY}, b::Vector)
    x .= A \ b
end

function LinearSolve._ldiv!(x::AbstractVector,
        A::Union{QR, LinearAlgebra.QRCompactWY}, b::AbstractVector)
    x .= A \ b
end

# Ambiguity removal
function LinearSolve._ldiv!(::SVector,
        A::Union{LinearAlgebra.QR, LinearAlgebra.QRCompactWY},
        b::AbstractVector)
    (A \ b)
end
function LinearSolve._ldiv!(::SVector,
        A::Union{LinearAlgebra.QR, LinearAlgebra.QRCompactWY},
        b::SVector)
    (A \ b)
end

@static if Base.USE_GPL_LIBS
    # SPQR and CHOLMOD Factor support
    function LinearSolve._ldiv!(x::Vector,
            A::Union{SparseArrays.SPQR.QRSparse, SparseArrays.CHOLMOD.Factor}, b::Vector)
        x .= A \ b
    end
    function LinearSolve._ldiv!(x::AbstractVector,
            A::Union{SparseArrays.SPQR.QRSparse, SparseArrays.CHOLMOD.Factor}, b::AbstractVector)
        x .= A \ b
    end
    function LinearSolve._ldiv!(::SVector,
            A::Union{SparseArrays.CHOLMOD.Factor, SparseArrays.SPQR.QRSparse},
            b::AbstractVector)
        (A \ b)
    end
    function LinearSolve._ldiv!(::SVector,
            A::Union{SparseArrays.CHOLMOD.Factor, SparseArrays.SPQR.QRSparse},
            b::SVector)
        (A \ b)
    end
end # @static if Base.USE_GPL_LIBS

function LinearSolve.pattern_changed(fact::Nothing, A::SparseArrays.SparseMatrixCSC)
    true
end

function LinearSolve.pattern_changed(fact, A::SparseArrays.SparseMatrixCSC)
    colptr0 = fact.colptr # has 0-based indices
    colptr1 = SparseArrays.getcolptr(A) # has 1-based indices
    length(colptr0) == length(colptr1) || return true
    @inbounds for i in eachindex(colptr0)
        colptr0[i] + 1 == colptr1[i] || return true
    end
    rowval0 = fact.rowval
    rowval1 = SparseArrays.getrowval(A)
    length(rowval0) == length(rowval1) || return true
    @inbounds for i in eachindex(rowval0)
        rowval0[i] + 1 == rowval1[i] || return true
    end
    return false
end

@static if Base.USE_GPL_LIBS
    function LinearSolve.defaultalg(
            A::AbstractSparseMatrixCSC{<:Union{Float64, ComplexF64}, Ti}, b,
            assump::OperatorAssumptions{Bool}) where {Ti}
        if assump.issq
            if length(b) <= 10_000 && length(nonzeros(A)) / length(A) < 2e-4
                LinearSolve.DefaultLinearSolver(LinearSolve.DefaultAlgorithmChoice.KLUFactorization)
            else
                LinearSolve.DefaultLinearSolver(LinearSolve.DefaultAlgorithmChoice.UMFPACKFactorization)
            end
        else
            LinearSolve.DefaultLinearSolver(LinearSolve.DefaultAlgorithmChoice.QRFactorization)
        end
    end
else
    function LinearSolve.defaultalg(
            A::AbstractSparseMatrixCSC{<:Union{Float64, ComplexF64}, Ti}, b,
            assump::OperatorAssumptions{Bool}) where {Ti}
        if assump.issq
            LinearSolve.DefaultLinearSolver(LinearSolve.DefaultAlgorithmChoice.KLUFactorization)
        elseif !assump.issq
            LinearSolve.DefaultLinearSolver(LinearSolve.DefaultAlgorithmChoice.QRFactorization)
        end
    end
end # @static if Base.USE_GPL_LIBS

# SPQR Handling
function LinearSolve.init_cacheval(
        alg::QRFactorization, A::AbstractSparseArray{<:Number, <:Integer}, b, u,
        Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions)
    nothing
end

function LinearSolve.init_cacheval(
        alg::QRFactorization, A::SparseMatrixCSC{Float64, <:Integer}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions)
    ArrayInterface.qr_instance(convert(AbstractMatrix, A), alg.pivot)
end

function LinearSolve.init_cacheval(
        alg::QRFactorization, A::Symmetric{<:Number, <:SparseMatrixCSC}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions)
    return nothing
end

LinearSolve.PrecompileTools.@compile_workload begin
    A = sprand(4, 4, 0.3) + I
    b = rand(4)
    prob = LinearProblem(A, b)
    sol = solve(prob, KLUFactorization())
    if Base.USE_GPL_LIBS
        sol = solve(prob, UMFPACKFactorization())
    end
end

end
