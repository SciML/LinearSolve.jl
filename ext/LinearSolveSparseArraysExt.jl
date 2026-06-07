module LinearSolveSparseArraysExt

using LinearSolve: LinearSolve, BLASELTYPES, pattern_changed, ArrayInterface,
    CHOLMODFactorization, GenericFactorization,
    GenericLUFactorization,
    KLUFactorization, PureKLUFactorization, LUFactorization,
    NormalCholeskyFactorization,
    OperatorAssumptions, LinearVerbosity,
    QRFactorization, RFLUFactorization, UMFPACKFactorization,
    SparseColumnPivotedQRFactorization, PersistentDropFactorization, solve
using SciMLOperators: AbstractSciMLOperator, has_concretization
using ArrayInterface: ArrayInterface
using LinearAlgebra: LinearAlgebra, I, Hermitian, Symmetric, cholesky, ldiv!, lu, lu!
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
# PureKLU (pure-Julia, no SuiteSparse) is a hard dependency and the default
# sparse LU; the SuiteSparse `KLUFactorization` above is unchanged.
import PureKLU
# SparseColumnPivotedQR (pure-Julia, rank-revealing column-pivoted sparse QR) is a
# hard dependency: the default sparse QR and the singular-LU fallback.
import SparseColumnPivotedQR
const SCPQR = SparseColumnPivotedQR
# Loading AMD activates SparseColumnPivotedQR's AMD extension, so its `:default`
# ordering resolves to AMD (1.5-2x faster factorization than natural ordering).
import AMD

LinearSolve.issparsematrixcsc(A::AbstractSparseMatrixCSC) = true
LinearSolve.issparsematrix(A::AbstractSparseArray) = true
function LinearSolve.make_SparseMatrixCSC(A::AbstractSparseArray)
    return SparseMatrixCSC(size(A)..., getcolptr(A), rowvals(A), nonzeros(A))
end
function LinearSolve.makeempty_SparseMatrixCSC(A::AbstractSparseArray)
    return SparseMatrixCSC(0, 0, [1], Int[], eltype(A)[])
end

function LinearSolve.init_cacheval(
        alg::RFLUFactorization,
        A::Union{AbstractSparseArray, LinearSolve.SciMLOperators.AbstractSciMLOperator}, b, u, Pl, Pr,
        maxiters::Int,
        abstol, reltol, verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return nothing, nothing
end

function LinearSolve.handle_sparsematrixcsc_lu(A::AbstractSparseMatrixCSC)
    return @static if Base.USE_GPL_LIBS
        lu(
            SparseMatrixCSC(size(A)..., getcolptr(A), rowvals(A), nonzeros(A)),
            check = false
        )
    else
        error("Sparse LU factorization requires GPL libraries (UMFPACK). Use `using Sparspak` for a non-GPL alternative or rebuild Julia with USE_GPL_LIBS=1")
    end
end

@static if Base.USE_GPL_LIBS
    function LinearSolve.defaultalg(
            A::Symmetric{<:BLASELTYPES, <:SparseMatrixCSC}, b, ::OperatorAssumptions{Bool}
        )
        LinearSolve.DefaultLinearSolver(LinearSolve.DefaultAlgorithmChoice.CHOLMODFactorization)
    end
else
    function LinearSolve.defaultalg(
            A::Symmetric{<:BLASELTYPES, <:SparseMatrixCSC}, b, ::OperatorAssumptions{Bool}
        )
        LinearSolve.DefaultLinearSolver(LinearSolve.DefaultAlgorithmChoice.CholeskyFactorization)
    end
end # @static if Base.USE_GPL_LIBS

function LinearSolve.defaultalg(
        A::AbstractSparseMatrixCSC{Tv, Ti}, b,
        assump::OperatorAssumptions{Bool}
    ) where {Tv, Ti}
    ext = Base.get_extension(LinearSolve, :LinearSolveSparspakExt)
    return if assump.issq && ext !== nothing
        LinearSolve.DefaultLinearSolver(LinearSolve.DefaultAlgorithmChoice.SparspakFactorization)
    elseif !assump.issq
        error("Generic number sparse factorization for non-square is not currently handled")
    elseif ext === nothing
        error("SparspakFactorization required for general sparse matrix types and with general Julia number types. Do `using Sparspak` in order to enable this functionality")
    else
        error("Unreachable reached. Please report this error with a reproducer.")
    end
end

function LinearSolve.init_cacheval(
        alg::GenericFactorization,
        A::Union{
            Hermitian{T, <:SparseMatrixCSC},
            Symmetric{T, <:SparseMatrixCSC},
        }, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    ) where {T}
    newA = copy(convert(AbstractMatrix, A))
    return LinearSolve.do_factorization(alg, newA, b, u)
end

@static if Base.USE_GPL_LIBS
    const PREALLOCATED_UMFPACK = SparseArrays.UMFPACK.UmfpackLU(
        SparseMatrixCSC(
            0, 0, [1],
            Int[], Float64[]
        )
    )
end # @static if Base.USE_GPL_LIBS

function LinearSolve.init_cacheval(
        alg::LUFactorization, A::AbstractSparseArray{<:Number, <:Integer}, b, u,
        Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return nothing
end

function LinearSolve.init_cacheval(
        alg::GenericLUFactorization, A::AbstractSparseArray{<:Number, <:Integer}, b, u,
        Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return nothing
end

function LinearSolve.init_cacheval(
        alg::UMFPACKFactorization, A::AbstractArray, b, u,
        Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return nothing
end

@static if Base.USE_GPL_LIBS
    function LinearSolve.init_cacheval(
            alg::LUFactorization, A::AbstractSparseArray{Float64, Int64}, b, u,
            Pl, Pr,
            maxiters::Int, abstol, reltol,
            verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
        )
        PREALLOCATED_UMFPACK
    end
    function LinearSolve.init_cacheval(
            alg::LUFactorization, A::AbstractSparseArray{T, Int64}, b, u,
            Pl, Pr,
            maxiters::Int, abstol, reltol,
            verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
        ) where {T <: BLASELTYPES}
        if LinearSolve.is_cusparse(A)
            LinearSolve.cudss_loaded(A) ? ArrayInterface.lu_instance(A) : nothing
        else
            SparseArrays.UMFPACK.UmfpackLU(
                SparseMatrixCSC{T, Int64}(
                    zero(Int64), zero(Int64), [Int64(1)], Int64[], T[]
                )
            )
        end
    end
    function LinearSolve.init_cacheval(
            alg::LUFactorization, A::AbstractSparseArray{T, Int32}, b, u,
            Pl, Pr,
            maxiters::Int, abstol, reltol,
            verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
        ) where {T <: BLASELTYPES}
        if LinearSolve.is_cusparse(A)
            LinearSolve.cudss_loaded(A) ? ArrayInterface.lu_instance(A) : nothing
        else
            SparseArrays.UMFPACK.UmfpackLU(
                SparseMatrixCSC{T, Int32}(
                    zero(Int32), zero(Int32), [Int32(1)], Int32[], T[]
                )
            )
        end
    end
end # @static if Base.USE_GPL_LIBS

function LinearSolve.init_cacheval(
        alg::LUFactorization, A::LinearSolve.GPUArraysCore.AnyGPUArray, b, u,
        Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return ArrayInterface.lu_instance(A)
end

function LinearSolve.init_cacheval(
        alg::UMFPACKFactorization, A::LinearSolve.GPUArraysCore.AnyGPUArray, b, u,
        Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return nothing
end

@static if Base.USE_GPL_LIBS
    function LinearSolve.init_cacheval(
            alg::UMFPACKFactorization, A::AbstractSparseArray{Float64, Int}, b, u, Pl, Pr,
            maxiters::Int, abstol,
            reltol,
            verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
        )
        PREALLOCATED_UMFPACK
    end

    function LinearSolve.init_cacheval(
            alg::UMFPACKFactorization, A::AbstractSparseArray{T, Int64}, b, u,
            Pl, Pr,
            maxiters::Int, abstol, reltol,
            verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
        ) where {T <: BLASELTYPES}
        SparseArrays.UMFPACK.UmfpackLU(
            SparseMatrixCSC{T, Int64}(
                zero(Int64), zero(Int64), [Int64(1)], Int64[], T[]
            )
        )
    end

    function LinearSolve.init_cacheval(
            alg::UMFPACKFactorization, A::AbstractSparseArray{T, Int32}, b, u,
            Pl, Pr,
            maxiters::Int, abstol, reltol,
            verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
        ) where {T <: BLASELTYPES}
        SparseArrays.UMFPACK.UmfpackLU(
            SparseMatrixCSC{T, Int32}(
                zero(Int32), zero(Int32), [Int32(1)], Int32[], T[]
            )
        )
    end

    function SciMLBase.solve!(
            cache::LinearSolve.LinearCache, alg::UMFPACKFactorization; kwargs...
        )
        A = cache.A
        A = convert(AbstractMatrix, A)
        if cache.isfresh
            cacheval = LinearSolve.@get_cacheval(cache, :UMFPACKFactorization)
            if alg.reuse_symbolic
                # Caches the symbolic factorization: https://github.com/JuliaLang/julia/pull/33738
                if length(cacheval.nzval) != length(nonzeros(A)) || alg.check_pattern && pattern_changed(cacheval, A)
                    fact = lu(
                        SparseMatrixCSC(
                            size(A)..., getcolptr(A), rowvals(A),
                            nonzeros(A)
                        ),
                        check = false
                    )
                else
                    fact = lu!(
                        cacheval,
                        SparseMatrixCSC(
                            size(A)..., getcolptr(A), rowvals(A),
                            nonzeros(A)
                        ), check = false
                    )
                end
            else
                fact = lu(
                    SparseMatrixCSC(size(A)..., getcolptr(A), rowvals(A), nonzeros(A)),
                    check = false
                )
            end
            cache.cacheval = fact
            cache.isfresh = false
        end

        F = LinearSolve.@get_cacheval(cache, :UMFPACKFactorization)
        if F.status == UMFPACK_OK
            y = ldiv!(cache.u, F, cache.b)
            SciMLBase.build_linear_solution(
                alg, y, nothing, cache; retcode = ReturnCode.Success
            )
        else
            @SciMLMessage("Solver failed", cache.verbose, :solver_failure)
            SciMLBase.build_linear_solution(
                alg, cache.u, nothing, cache; retcode = ReturnCode.Infeasible
            )
        end
    end

else
    function SciMLBase.solve!(
            cache::LinearSolve.LinearCache, alg::UMFPACKFactorization; kwargs...
        )
        error("UMFPACKFactorization requires GPL libraries (UMFPACK). Rebuild Julia with USE_GPL_LIBS=1 or use an alternative algorithm like SparspakFactorization")
    end
end # @static if Base.USE_GPL_LIBS

function LinearSolve.init_cacheval(
        alg::KLUFactorization, A::AbstractArray, b, u, Pl,
        Pr,
        maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return nothing
end

function LinearSolve.init_cacheval(
        alg::KLUFactorization, A::LinearSolve.GPUArraysCore.AnyGPUArray, b, u,
        Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return nothing
end

const PREALLOCATED_KLU = KLU.KLUFactorization(
    SparseMatrixCSC(
        0, 0, [1], Int[],
        Float64[]
    )
)

function LinearSolve.init_cacheval(
        alg::KLUFactorization, A::AbstractSparseArray{Float64, Int64}, b, u, Pl, Pr,
        maxiters::Int, abstol,
        reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return PREALLOCATED_KLU
end

# KLU supports Float64 and ComplexF64 (KLUTypes)
function LinearSolve.init_cacheval(
        alg::KLUFactorization, A::AbstractSparseArray{T, Int64}, b, u, Pl, Pr,
        maxiters::Int, abstol,
        reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    ) where {T <: KLU.KLUTypes}
    return KLU.KLUFactorization(
        SparseMatrixCSC{T, Int64}(
            0, 0, [Int64(1)], Int64[], T[]
        )
    )
end

function LinearSolve.init_cacheval(
        alg::KLUFactorization, A::AbstractSparseArray{Float64, Int32}, b, u, Pl, Pr,
        maxiters::Int, abstol,
        reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return KLU.KLUFactorization(
        SparseMatrixCSC{Float64, Int32}(
            0, 0, [Int32(1)], Int32[], Float64[]
        )
    )
end

function LinearSolve.init_cacheval(
        alg::KLUFactorization, A::AbstractSparseArray{T, Int32}, b, u, Pl, Pr,
        maxiters::Int, abstol,
        reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    ) where {T <: KLU.KLUTypes}
    return KLU.KLUFactorization(
        SparseMatrixCSC{T, Int32}(
            0, 0, [Int32(1)], Int32[], T[]
        )
    )
end

# AbstractSciMLOperator handling for sparse factorizations
function LinearSolve.init_cacheval(
        alg::KLUFactorization, A::AbstractSciMLOperator, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    if has_concretization(A)
        return LinearSolve.init_cacheval(
            alg, convert(AbstractMatrix, A), b, u, Pl, Pr,
            maxiters, abstol, reltol, verbose, assumptions
        )
    else
        nothing
    end
end

function LinearSolve.init_cacheval(
        alg::UMFPACKFactorization, A::AbstractSciMLOperator, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    if has_concretization(A)
        return LinearSolve.init_cacheval(
            alg, convert(AbstractMatrix, A), b, u, Pl, Pr,
            maxiters, abstol, reltol, verbose, assumptions
        )
    else
        nothing
    end
end

function LinearSolve.init_cacheval(
        alg::CHOLMODFactorization, A::AbstractSciMLOperator, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    if has_concretization(A)
        return LinearSolve.init_cacheval(
            alg, convert(AbstractMatrix, A), b, u, Pl, Pr,
            maxiters, abstol, reltol, verbose, assumptions
        )
    else
        nothing
    end
end


function LinearSolve.init_cacheval(
        alg::NormalCholeskyFactorization, A::AbstractSciMLOperator, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    if has_concretization(A)
        return LinearSolve.init_cacheval(
            alg, convert(AbstractMatrix, A), b, u, Pl, Pr,
            maxiters, abstol, reltol, verbose, assumptions
        )
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
            if length(cacheval.nzval) != length(nonzeros(A)) || alg.check_pattern && pattern_changed(cacheval, A)
                fact = KLU.klu(
                    SparseMatrixCSC(
                        size(A)..., getcolptr(A), rowvals(A),
                        nonzeros(A)
                    ),
                    check = false
                )
            else
                fact = KLU.klu!(cacheval, nonzeros(A), check = false)
            end
        else
            # New fact each time since the sparsity pattern can change
            # and thus it needs to reallocate. `check = false` matches the
            # `reuse_symbolic = true` branch and keeps singular matrices from
            # throwing `LinearAlgebra.SingularException`; the status check
            # below maps that to `ReturnCode.Infeasible` instead. Fixes
            # https://github.com/SciML/LinearSolve.jl/issues/991.
            fact = KLU.klu(
                SparseMatrixCSC(
                    size(A)..., getcolptr(A), rowvals(A),
                    nonzeros(A)
                ),
                check = false
            )
        end
        cache.cacheval = fact
        cache.isfresh = false
    end
    F = LinearSolve.@get_cacheval(cache, :KLUFactorization)
    return if F.common.status == KLU.KLU_OK
        y = ldiv!(cache.u, F, cache.b)
        if all(isfinite, y)
            SciMLBase.build_linear_solution(
                alg, y, nothing, cache; retcode = ReturnCode.Success
            )
        else
            # KLU can report `KLU_OK` on a numerically singular matrix (a
            # tiny-but-nonzero pivot, common when explicit stored zeros mask a
            # rank deficiency) yet produce non-finite output. Surface that as a
            # failure instead of a silent NaN `Success`, matching the default
            # solver's finiteness check.
            @SciMLMessage(
                "Solver produced non-finite values; matrix is likely singular",
                cache.verbose, :solver_failure
            )
            SciMLBase.build_linear_solution(
                alg, cache.u, nothing, cache; retcode = ReturnCode.Infeasible
            )
        end
    else
        @SciMLMessage("Solver failed", cache.verbose, :solver_failure)
        SciMLBase.build_linear_solution(
            alg, cache.u, nothing, cache; retcode = ReturnCode.Infeasible
        )
    end
end

# --- PureKLU: pure-Julia KLU, the default sparse LU (no SuiteSparse dependency) ---

function LinearSolve.init_cacheval(
        alg::PureKLUFactorization, A::AbstractArray, b, u, Pl,
        Pr,
        maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return nothing
end

function LinearSolve.init_cacheval(
        alg::PureKLUFactorization, A::LinearSolve.GPUArraysCore.AnyGPUArray, b, u,
        Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return nothing
end

const PREALLOCATED_PUREKLU = PureKLU.KLUFactorization(
    SparseMatrixCSC(
        0, 0, [1], Int[],
        Float64[]
    )
)

function LinearSolve.init_cacheval(
        alg::PureKLUFactorization, A::AbstractSparseArray{Float64, Int64}, b, u, Pl, Pr,
        maxiters::Int, abstol,
        reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return PREALLOCATED_PUREKLU
end

function LinearSolve.init_cacheval(
        alg::PureKLUFactorization, A::AbstractSparseArray{T, Int64}, b, u, Pl, Pr,
        maxiters::Int, abstol,
        reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    ) where {T <: Union{Float64, ComplexF64}}
    return PureKLU.KLUFactorization(
        SparseMatrixCSC{T, Int64}(
            0, 0, [Int64(1)], Int64[], T[]
        )
    )
end

function LinearSolve.init_cacheval(
        alg::PureKLUFactorization, A::AbstractSparseArray{Float64, Int32}, b, u, Pl, Pr,
        maxiters::Int, abstol,
        reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return PureKLU.KLUFactorization(
        SparseMatrixCSC{Float64, Int32}(
            0, 0, [Int32(1)], Int32[], Float64[]
        )
    )
end

function LinearSolve.init_cacheval(
        alg::PureKLUFactorization, A::AbstractSparseArray{T, Int32}, b, u, Pl, Pr,
        maxiters::Int, abstol,
        reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    ) where {T <: Union{Float64, ComplexF64}}
    return PureKLU.KLUFactorization(
        SparseMatrixCSC{T, Int32}(
            0, 0, [Int32(1)], Int32[], T[]
        )
    )
end

function LinearSolve.init_cacheval(
        alg::PureKLUFactorization, A::AbstractSciMLOperator, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    if has_concretization(A)
        return LinearSolve.init_cacheval(
            alg, convert(AbstractMatrix, A), b, u, Pl, Pr,
            maxiters, abstol, reltol, verbose, assumptions
        )
    else
        nothing
    end
end

function SciMLBase.solve!(
        cache::LinearSolve.LinearCache, alg::PureKLUFactorization; kwargs...
    )
    A = cache.A
    A = convert(AbstractMatrix, A)
    if cache.isfresh
        # PureKLU occupies the default polyalgorithm's `:KLUFactorization` slot
        # (the default's KLU choice resolves to PureKLU), so it reads that field.
        cacheval = LinearSolve.@get_cacheval(cache, :KLUFactorization)
        if alg.reuse_symbolic
            if length(cacheval.nzval) != length(nonzeros(A)) ||
                    alg.check_pattern && pattern_changed(cacheval, A)
                fact = PureKLU.klu(
                    SparseMatrixCSC(
                        size(A)..., getcolptr(A), rowvals(A),
                        nonzeros(A)
                    ),
                    check = false, use_fma = alg.use_fma,
                    fully_preallocated = alg.fully_preallocated
                )
            else
                fact = PureKLU.klu!(cacheval, nonzeros(A), check = false)
            end
        else
            # New fact each time since the sparsity pattern can change and thus
            # it needs to reallocate. `check = false` keeps singular matrices from
            # throwing; the status check below maps that to `ReturnCode.Infeasible`.
            fact = PureKLU.klu(
                SparseMatrixCSC(
                    size(A)..., getcolptr(A), rowvals(A),
                    nonzeros(A)
                ),
                check = false, use_fma = alg.use_fma,
                fully_preallocated = alg.fully_preallocated
            )
        end
        cache.cacheval = fact
        cache.isfresh = false
    end
    F = LinearSolve.@get_cacheval(cache, :KLUFactorization)
    return if F.common.status == PureKLU.KLU_OK
        y = ldiv!(cache.u, F, cache.b)
        if all(isfinite, y)
            SciMLBase.build_linear_solution(
                alg, y, nothing, cache; retcode = ReturnCode.Success
            )
        else
            # PureKLU (like SuiteSparse KLU) can report `KLU_OK` on a numerically
            # singular matrix (a tiny-but-nonzero pivot, common when explicit
            # stored zeros mask a rank deficiency) yet produce non-finite output.
            # Surface that as a failure instead of a silent NaN `Success`,
            # matching the default solver's finiteness check.
            @SciMLMessage(
                "Solver produced non-finite values; matrix is likely singular",
                cache.verbose, :solver_failure
            )
            SciMLBase.build_linear_solution(
                alg, cache.u, nothing, cache; retcode = ReturnCode.Infeasible
            )
        end
    else
        @SciMLMessage("Solver failed", cache.verbose, :solver_failure)
        SciMLBase.build_linear_solution(
            alg, cache.u, nothing, cache; retcode = ReturnCode.Infeasible
        )
    end
end

# --- SparseColumnPivotedQR: pure-Julia rank-revealing column-pivoted sparse QR ---
# The default sparse QR (non-square sparse systems) and the singular-LU fallback.

# One preallocated factorization per supported element type. They give the
# default solver's cacheval slot a concrete element type so the singular-LU
# fallback can store its factorization into it type-stably.
const PREALLOCATED_SCPQR_F64 = SCPQR.scpqr(sparse(reshape([1.0], 1, 1)))
const PREALLOCATED_SCPQR_C64 = SCPQR.scpqr(sparse(reshape([ComplexF64(1)], 1, 1)))

function LinearSolve.init_cacheval(
        alg::SparseColumnPivotedQRFactorization, A::AbstractArray, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return nothing
end

function LinearSolve.init_cacheval(
        alg::SparseColumnPivotedQRFactorization, A::LinearSolve.GPUArraysCore.AnyGPUArray,
        b, u, Pl, Pr, maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return nothing
end

function LinearSolve.init_cacheval(
        alg::SparseColumnPivotedQRFactorization, A::AbstractSparseArray{Float64, <:Integer},
        b, u, Pl, Pr, maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return PREALLOCATED_SCPQR_F64
end

function LinearSolve.init_cacheval(
        alg::SparseColumnPivotedQRFactorization, A::AbstractSparseArray{ComplexF64, <:Integer},
        b, u, Pl, Pr, maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return PREALLOCATED_SCPQR_C64
end

function LinearSolve.init_cacheval(
        alg::SparseColumnPivotedQRFactorization, A::AbstractSciMLOperator, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    if has_concretization(A)
        return LinearSolve.init_cacheval(
            alg, convert(AbstractMatrix, A), b, u, Pl, Pr,
            maxiters, abstol, reltol, verbose, assumptions
        )
    else
        nothing
    end
end

function SciMLBase.solve!(
        cache::LinearSolve.LinearCache, alg::SparseColumnPivotedQRFactorization; kwargs...
    )
    A = cache.A
    A = convert(AbstractMatrix, A)
    if cache.isfresh
        cacheval = LinearSolve.@get_cacheval(cache, :SparseColumnPivotedQRFactorization)
        Acsc = convert(SparseMatrixCSC, A)
        # Reuse the cached factorization's symbolic analysis + workspace when the
        # shape matches (it re-analyzes internally if the sparsity pattern changed);
        # otherwise factor fresh. The preallocated factorization has a different
        # shape, so the first real solve always factors fresh.
        fact = if alg.reuse_symbolic && cacheval isa SCPQR.SparseColumnPivotedQRFactorization &&
                size(cacheval) == size(A)
            SCPQR.scpqr_refactor!(cacheval, Acsc)
        else
            SCPQR.scpqr(Acsc; ordering = alg.ordering)
        end
        cache.cacheval = fact
        cache.isfresh = false
    end
    F = LinearSolve.@get_cacheval(cache, :SparseColumnPivotedQRFactorization)
    y = ldiv!(cache.u, F, cache.b)
    return SciMLBase.build_linear_solution(
        alg, y, nothing, cache; retcode = ReturnCode.Success
    )
end

# Build a column-pivoted sparse QR factorization for the default sparse-LU
# singular fallback (`_do_sparse_qr_fallback` in src/default.jl).
function LinearSolve.sparse_colpivqr_factorize(A)
    return SCPQR.scpqr(convert(SparseMatrixCSC, convert(AbstractMatrix, A)))
end

function LinearSolve.init_cacheval(
        alg::CHOLMODFactorization,
        A::AbstractArray, b, u,
        Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return nothing
end

@static if Base.USE_GPL_LIBS
    const PREALLOCATED_CHOLMOD = cholesky(sparse(reshape([1.0], 1, 1)))

    function LinearSolve.init_cacheval(
            alg::CHOLMODFactorization,
            A::Union{SparseMatrixCSC{T, Int}, Symmetric{T, SparseMatrixCSC{T, Int}}}, b, u,
            Pl, Pr,
            maxiters::Int, abstol, reltol,
            verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
        ) where {
            T <:
            Float64,
        }
        PREALLOCATED_CHOLMOD
    end

    function LinearSolve.init_cacheval(
            alg::CHOLMODFactorization,
            A::Union{SparseMatrixCSC{T, Int}, Symmetric{T, SparseMatrixCSC{T, Int}}}, b, u,
            Pl, Pr,
            maxiters::Int, abstol, reltol,
            verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
        ) where {
            T <:
            BLASELTYPES,
        }
        cholesky(sparse(reshape([one(T)], 1, 1)))
    end
end # @static if Base.USE_GPL_LIBS

function LinearSolve.init_cacheval(
        alg::NormalCholeskyFactorization,
        A::Union{
            AbstractSparseArray{T}, LinearSolve.GPUArraysCore.AnyGPUArray,
            Symmetric{T, <:AbstractSparseArray{T}},
        }, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    ) where {T <: BLASELTYPES}
    return if LinearSolve.is_cusparse_csc(A)
        nothing
    elseif LinearSolve.is_cusparse_csr(A) && !LinearSolve.cudss_loaded(A)
        nothing
    else
        ArrayInterface.cholesky_instance(convert(AbstractMatrix, A))
    end
end

# Ambiguity removal
function LinearSolve._ldiv!(
        ::SVector,
        A::Union{LinearAlgebra.QR, LinearAlgebra.QRCompactWY},
        b::AbstractVector
    )
    return (A \ b)
end
function LinearSolve._ldiv!(
        ::SVector,
        A::Union{LinearAlgebra.QR, LinearAlgebra.QRCompactWY},
        b::SVector
    )
    return (A \ b)
end

@static if Base.USE_GPL_LIBS
    # ldiv!() for CHOLMOD was added in 1.12: https://github.com/JuliaSparse/SparseArrays.jl/pull/547
    @static if VERSION < v"1.12"
        function LinearSolve._ldiv!(
                x::Vector,
                A::SparseArrays.CHOLMOD.Factor, b::Vector
            )
            x .= A \ b
        end
        function LinearSolve._ldiv!(
                x::AbstractVector,
                A::SparseArrays.CHOLMOD.Factor, b::AbstractVector
            )
            x .= A \ b
        end
    end

    # ldiv!() for SPQR should be in Julia 1.13: https://github.com/JuliaSparse/SparseArrays.jl/pull/676
    function LinearSolve._ldiv!(
            x::Vector,
            A::SparseArrays.SPQR.QRSparse, b::Vector
        )
        x .= A \ b
    end
    function LinearSolve._ldiv!(
            x::AbstractVector,
            A::SparseArrays.SPQR.QRSparse, b::AbstractVector
        )
        x .= A \ b
    end

    function LinearSolve._ldiv!(
            ::SVector,
            A::Union{SparseArrays.CHOLMOD.Factor, SparseArrays.SPQR.QRSparse},
            b::AbstractVector
        )
        (A \ b)
    end
    function LinearSolve._ldiv!(
            ::SVector,
            A::Union{SparseArrays.CHOLMOD.Factor, SparseArrays.SPQR.QRSparse},
            b::SVector
        )
        (A \ b)
    end
end # @static if Base.USE_GPL_LIBS

function LinearSolve.pattern_changed(
        fact::Nothing,
        A::SparseArrays.AbstractSparseMatrixCSC{<:Any, <:Integer}
    )
    return true
end

function LinearSolve.pattern_changed(fact, A::SparseArrays.AbstractSparseMatrixCSC{<:Any, <:Integer})
    colptr0 = fact.colptr # has 0-based indices
    colptr1 = SparseArrays.getcolptr(A) # has 1-based indices
    length(colptr0) == length(colptr1) || return true
    @inbounds for i in eachindex(colptr0)
        colptr0[i] + 1 == colptr1[i] || return true
    end
    rowval0 = fact.rowval
    rowval1 = SparseArrays.rowvals(A)
    length(rowval0) == length(rowval1) || return true
    @inbounds for i in eachindex(rowval0)
        rowval0[i] + 1 == rowval1[i] || return true
    end
    return false
end

# Heuristic shared by the sparse default's LU and QR branches. `true` (small, or
# medium and very sparse / "less structure") favors the pure-Julia KLU-style
# solvers: PureKLU for LU and SparseColumnPivotedQR for QR. `false` ("more
# structure") favors the SuiteSparse solvers: UMFPACK for LU and SPQR for QR.
# The `length(b) <= 1_000` branch is the "small enough" fast path; the second is
# the "medium and sufficiently sparse" rule.
function LinearSolve.use_klulike_sparse_structure(A::AbstractSparseMatrixCSC, b)
    return length(b) <= 1_000 ||
        (length(b) <= 10_000 && length(nonzeros(A)) / length(A) < 2.0e-4)
end

@static if Base.USE_GPL_LIBS
    function LinearSolve.defaultalg(
            A::AbstractSparseMatrixCSC{<:Union{Float64, ComplexF64}, Ti}, b,
            assump::OperatorAssumptions{Bool}
        ) where {Ti}
        klulike = LinearSolve.use_klulike_sparse_structure(A, b)
        if assump.issq
            # Less structure → PureKLU (pure-Julia); more structure → UMFPACK.
            if klulike
                LinearSolve.DefaultLinearSolver(LinearSolve.DefaultAlgorithmChoice.KLUFactorization)
            else
                LinearSolve.DefaultLinearSolver(LinearSolve.DefaultAlgorithmChoice.UMFPACKFactorization)
            end
        else
            # Same split for sparse QR: less structure → SparseColumnPivotedQR
            # (pure-Julia); more structure → SuiteSparse SPQR (`QRFactorization`).
            if klulike
                LinearSolve.DefaultLinearSolver(LinearSolve.DefaultAlgorithmChoice.SparseColumnPivotedQRFactorization)
            else
                LinearSolve.DefaultLinearSolver(LinearSolve.DefaultAlgorithmChoice.QRFactorization)
            end
        end
    end
else
    function LinearSolve.defaultalg(
            A::AbstractSparseMatrixCSC{<:Union{Float64, ComplexF64}, Ti}, b,
            assump::OperatorAssumptions{Bool}
        ) where {Ti}
        # No SuiteSparse (UMFPACK/SPQR) available: always use the pure-Julia solvers.
        if assump.issq
            LinearSolve.DefaultLinearSolver(LinearSolve.DefaultAlgorithmChoice.KLUFactorization)
        else
            LinearSolve.DefaultLinearSolver(LinearSolve.DefaultAlgorithmChoice.SparseColumnPivotedQRFactorization)
        end
    end
end # @static if Base.USE_GPL_LIBS

# SPQR Handling
function LinearSolve.init_cacheval(
        alg::QRFactorization, A::AbstractSparseArray{<:Number, <:Integer}, b, u,
        Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return nothing
end

function LinearSolve.init_cacheval(
        alg::QRFactorization, A::SparseMatrixCSC{Float64, <:Integer}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    )
    return ArrayInterface.qr_instance(convert(AbstractMatrix, A), alg.pivot)
end

function LinearSolve.init_cacheval(
        alg::QRFactorization, A::SparseMatrixCSC{ComplexF64, <:Integer}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    )
    return ArrayInterface.qr_instance(convert(AbstractMatrix, A), alg.pivot)
end

function LinearSolve.init_cacheval(
        alg::QRFactorization, A::Symmetric{<:Number, <:SparseMatrixCSC}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    )
    return nothing
end

LinearSolve.PrecompileTools.@compile_workload begin
    A = sprand(4, 4, 0.3) + I
    b = rand(4)
    prob = LinearProblem(A, b)
    sol = solve(prob, PureKLUFactorization())
    sol = solve(prob, KLUFactorization())
    sol = solve(prob, SparseColumnPivotedQRFactorization())
    if Base.USE_GPL_LIBS
        sol = solve(prob, UMFPACKFactorization())
    end
end

# --- PersistentDropFactorization: drop only persistently-zero stored entries ---
# State carried in the cacheval. The reduced matrix is the operand fed to an inner
# LinearCache (any sparse algorithm, or the default polyalgorithm). Because the
# inner cache solves the reduced matrix, the inner default's singular LU → QR
# fallback factorizes the reduced matrix too -- the reduction flows to QR for free.
mutable struct PersistentDropCache{Tinner, Tv, Ti}
    colptr::Vector{Ti}              # fixed FULL stored pattern (reference)
    rowval::Vector{Ti}
    n::Int
    mask::Vector{Bool}              # over full nnz positions: kept (ever-nonzero)?
    keep::Vector{Int}              # kept positions into the full nzval (sorted, CSC order)
    reduced::SparseMatrixCSC{Tv, Ti} # operand; nzval overwritten in place each step
    inner::Tinner                   # inner LinearCache solving the reduced matrix
    nanalyze::Int                   # number of (re)analyses (pattern widenings + initial)
    nrefactor::Int                  # number of numeric-only refactors
end

# Build the reduced (colptr, rowval, nzval) and the keep list from the full
# pattern + mask, gathering current values. Allocates a fresh reduced matrix.
function _persistent_reduced(
        colptr::Vector{Ti}, rowval, n, mask, nzval::AbstractVector{Tv}
    ) where {Ti, Tv}
    keep = Int[]
    rcolptr = Vector{Ti}(undef, n + 1)
    rrowval = Ti[]
    rnzval = Tv[]
    @inbounds for j in 1:n
        rcolptr[j] = length(rrowval) + 1
        for p in colptr[j]:(colptr[j + 1] - 1)
            if mask[p]
                push!(keep, p)
                push!(rrowval, rowval[p])
                push!(rnzval, nzval[p])
            end
        end
    end
    rcolptr[n + 1] = length(rrowval) + 1
    return keep, SparseMatrixCSC(n, n, rcolptr, rrowval, rnzval)
end

function LinearSolve.init_cacheval(
        alg::PersistentDropFactorization,
        A::AbstractSparseMatrixCSC, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    n = size(A, 1)
    colptr = copy(getcolptr(A))
    rowval = copy(rowvals(A))
    nz = nonzeros(A)
    mask = alg.drop_initial_zeros ? Bool[!iszero(v) for v in nz] : trues(length(nz))
    keep, reduced = _persistent_reduced(colptr, rowval, n, mask, nz)
    inner_alg = alg.alg === nothing ? LinearSolve.defaultalg(reduced, b, assumptions) : alg.alg
    inner = SciMLBase.init(
        LinearProblem(reduced, b), inner_alg;
        assumptions = assumptions,
        alias = LinearSolve.LinearAliasSpecifier(alias_A = true, alias_b = true),
        maxiters = maxiters, abstol = abstol, reltol = reltol, verbose = verbose
    )
    return PersistentDropCache(colptr, rowval, n, mask, keep, reduced, inner, 1, 0)
end

# Non-sparse input: PersistentDrop has nothing to do, so it is an error to use it
# off the sparse path (it is a sparse-only meta-algorithm).
function LinearSolve.init_cacheval(
        alg::PersistentDropFactorization, A, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    throw(ArgumentError("PersistentDropFactorization requires a sparse (CSC) matrix"))
end

function SciMLBase.solve!(
        cache::LinearSolve.LinearCache, alg::PersistentDropFactorization; kwargs...
    )
    st = cache.cacheval::PersistentDropCache
    A = cache.A
    nz = nonzeros(A)
    length(nz) == length(st.mask) ||
        throw(ArgumentError("PersistentDropFactorization: the stored sparsity pattern \
                             must be constant across solves (stored nnz changed)"))
    # Widen the union mask if any persistently-dead entry has just activated.
    grew = false
    @inbounds for i in eachindex(nz)
        if !st.mask[i] && !iszero(nz[i])
            st.mask[i] = true
            grew = true
        end
    end
    if grew
        # Pattern grew: rebuild the reduced operand (new pattern). Handing the inner
        # cache a new-pattern matrix makes it re-analyze (and clears any QR fallback).
        st.keep, st.reduced = _persistent_reduced(st.colptr, st.rowval, st.n, st.mask, nz)
        st.inner.A = st.reduced
        st.nanalyze += 1
    else
        # Stable pattern: gather current values into the existing reduced operand and
        # hand it back so the inner cache reuses its symbolic analysis (numeric refactor).
        rnz = nonzeros(st.reduced)
        keep = st.keep
        @inbounds @simd for j in eachindex(keep)
            rnz[j] = nz[keep[j]]
        end
        st.inner.A = st.reduced
        st.nrefactor += 1
    end
    st.inner.b = cache.b
    sol = SciMLBase.solve!(st.inner)
    copyto!(cache.u, sol.u)
    cache.isfresh = false
    return SciMLBase.build_linear_solution(
        alg, cache.u, nothing, cache;
        retcode = sol.retcode, iters = sol.iters, stats = sol.stats
    )
end

end
