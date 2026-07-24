module LinearSolveSparseArraysExt

using LinearSolve: LinearSolve, BLASELTYPES, pattern_changed, ArrayInterface,
    CHOLMODFactorization, GenericFactorization,
    GenericLUFactorization,
    KLUFactorization, PureKLUFactorization, LUFactorization,
    NormalCholeskyFactorization,
    OperatorAssumptions, LinearVerbosity,
    QRFactorization, RFLUFactorization, UMFPACKFactorization,
    SparseColumnPivotedQRFactorization, SupernodalLUFactorization, solve
using SciMLOperators: AbstractSciMLOperator, has_concretization
using ArrayInterface: ArrayInterface
using LinearAlgebra: LinearAlgebra, I, Hermitian, Symmetric, cholesky, ldiv!, lu, lu!
using SparseArrays: SparseArrays, AbstractSparseArray, AbstractSparseMatrixCSC,
    SparseMatrixCSC,
    nonzeros, rowvals, getcolptr, sparse, sprand, dropzeros!, nnz
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
# SupernodalLU (pure-Julia supernodal left-right-looking LU, Schenk-Gärtner
# method) is vendored in src/SupernodalLU: the BLAS-3 sparse LU for
# structured systems.
const SNLU = LinearSolve.SupernodalLU
# SparseColumnPivotedQR (pure-Julia, rank-revealing column-pivoted sparse QR) is a
# hard dependency: the default sparse QR and the singular-LU fallback.
import SparseColumnPivotedQR
const SCPQR = SparseColumnPivotedQR
# Loading AMD activates SparseColumnPivotedQR's AMD extension, so its `:default`
# ordering resolves to AMD (1.5-2x faster factorization than natural ordering).
import AMD

LinearSolve.issparsematrixcsc(A::AbstractSparseMatrixCSC) = true
LinearSolve.issparsematrix(A::AbstractSparseArray) = true
LinearSolve.make_SparseMatrixCSC(A::SparseMatrixCSC) = A
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
    # PureKLU is a pure-Julia, hard dependency that factors any `Number` element
    # type, so it is the default sparse LU for generic (non-BLAS) eltypes such as
    # BigFloat — no `using Sparspak` required. Unlike Sparspak's symbolic reuse,
    # PureKLU re-analyzes when the sparsity pattern changes across solves (e.g. the
    # per-solve dropzeros of the nonstructural_zeros path), which previously
    # produced invalid factorizations and stalled BVP Newton iterations. A
    # (near-)singular matrix falls back to the generic column-pivoted sparse QR via
    # the default polyalgorithm's sparse-LU fallback chain.
    return if assump.issq
        LinearSolve.DefaultLinearSolver(LinearSolve.DefaultAlgorithmChoice.KLUFactorization)
    else
        error("Generic number sparse factorization for non-square is not currently handled")
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
        A = LinearSolve.reduce_operand!(cache.sparse_reduction, A)
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
                alg, y, nothing, nothing; retcode = ReturnCode.Success
            )
        else
            @SciMLMessage("Solver failed", cache.verbose, :solver_failure)
            SciMLBase.build_linear_solution(
                alg, cache.u, nothing, nothing; retcode = ReturnCode.Infeasible
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
    # Drop persistent nonstructural zeros when the assumption requests it; a no-op
    # (returns `A`) for the default solver and when no reduction is active.
    A = LinearSolve.reduce_operand!(cache.sparse_reduction, A)
    A = convert(AbstractMatrix, A)
    if cache.isfresh
        cacheval = LinearSolve.@get_cacheval(cache, :KLUFactorization)
        if alg.reuse_symbolic
            if length(cacheval.nzval) != length(nonzeros(A)) || alg.check_pattern && pattern_changed(cacheval, A)
                fact = KLU.klu(
                    LinearSolve.make_SparseMatrixCSC(A),
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
                LinearSolve.make_SparseMatrixCSC(A),
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
                alg, y, nothing, nothing; retcode = ReturnCode.Success
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
                alg, cache.u, nothing, nothing; retcode = ReturnCode.Infeasible
            )
        end
    else
        @SciMLMessage("Solver failed", cache.verbose, :solver_failure)
        SciMLBase.build_linear_solution(
            alg, cache.u, nothing, nothing; retcode = ReturnCode.Infeasible
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

# Generic element types (e.g. BigFloat): PureKLU is pure-Julia and factors any
# `Number` element type, so it serves as the default sparse LU for non-BLAS
# eltypes too (replacing Sparspak in the default polyalgorithm). The empty
# cacheval carries the correct element type so `klu!`/`klu` dispatch is concrete.
function LinearSolve.init_cacheval(
        alg::PureKLUFactorization, A::AbstractSparseArray{T, Ti}, b, u, Pl, Pr,
        maxiters::Int, abstol,
        reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    ) where {T <: Number, Ti <: Integer}
    return PureKLU.KLUFactorization(
        SparseMatrixCSC{T, Ti}(
            0, 0, [one(Ti)], Ti[], T[]
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
    A = LinearSolve.reduce_operand!(cache.sparse_reduction, A)
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
                alg, y, nothing, nothing; retcode = ReturnCode.Success
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
                alg, cache.u, nothing, nothing; retcode = ReturnCode.Infeasible
            )
        end
    else
        @SciMLMessage("Solver failed", cache.verbose, :solver_failure)
        SciMLBase.build_linear_solution(
            alg, cache.u, nothing, nothing; retcode = ReturnCode.Infeasible
        )
    end
end

# --- SupernodalLU: pure-Julia supernodal left-right-looking LU (Schenk-Gärtner) ---
# The BLAS-3 sparse LU for structured (PDE-mesh-like) systems (vendored in src/SupernodalLU).

function LinearSolve.init_cacheval(
        alg::SupernodalLUFactorization, A::AbstractArray, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return nothing
end

function LinearSolve.init_cacheval(
        alg::SupernodalLUFactorization, A::LinearSolve.GPUArraysCore.AnyGPUArray, b, u,
        Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return nothing
end

const PREALLOCATED_SUPERNODAL = SNLU.snlu(
    SparseMatrixCSC{Float64, Int64}(0, 0, [Int64(1)], Int64[], Float64[])
)

function LinearSolve.init_cacheval(
        alg::SupernodalLUFactorization, A::AbstractSparseArray{Float64, Int64}, b, u,
        Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return PREALLOCATED_SUPERNODAL
end

# SupernodalLU is pure Julia and factors any `Number` element type; the empty
# cacheval carries the correct types so `pplu!`/`pplu` dispatch is concrete.
function LinearSolve.init_cacheval(
        alg::SupernodalLUFactorization, A::AbstractSparseArray{T, Ti}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    ) where {T <: Number, Ti <: Integer}
    return SNLU.snlu(
        SparseMatrixCSC{T, Ti}(0, 0, [one(Ti)], Ti[], T[])
    )
end

function LinearSolve.init_cacheval(
        alg::SupernodalLUFactorization, A::AbstractSciMLOperator, b, u, Pl, Pr,
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

function LinearSolve.pattern_changed(
        F::SNLU.SupernodalLUFactor, A::SparseArrays.AbstractSparseMatrixCSC
    )
    Aold = F.A
    return getcolptr(Aold) != getcolptr(A) || rowvals(Aold) != rowvals(A)
end

function SciMLBase.solve!(
        cache::LinearSolve.LinearCache, alg::SupernodalLUFactorization; kwargs...
    )
    A = cache.A
    A = LinearSolve.reduce_operand!(cache.sparse_reduction, A)
    A = convert(AbstractMatrix, A)
    if cache.isfresh
        # SupernodalLU occupies the default polyalgorithm's `:UMFPACKFactorization`
        # slot (the default's UMFPACK choice resolves to it), so it reads that
        # field; for standalone use the symbol is ignored.
        cacheval = LinearSolve.@get_cacheval(cache, :UMFPACKFactorization)
        As = SparseMatrixCSC(size(A)..., getcolptr(A), rowvals(A), nonzeros(A))
        if alg.reuse_symbolic && size(cacheval) == size(As) &&
                nnz(cacheval.A) == nnz(As) &&
                !(alg.check_pattern && pattern_changed(cacheval, As))
            # numeric-only refactorization: reuses the analysis, matching, and
            # all numeric storage (allocation-free)
            fact = SNLU.snlu!(cacheval, As)
        else
            # `check = false`: static pivoting never aborts — numerically
            # singular systems surface through the finiteness check below.
            fact = SNLU.snlu(
                As; ordering = alg.ordering, matching = alg.matching,
                eps_pivot = alg.eps_pivot, threaded = alg.threaded,
                dense_alg = alg.dense_alg, check = false
            )
        end
        cache.cacheval = fact
        cache.isfresh = false
    end
    F = LinearSolve.@get_cacheval(cache, :UMFPACKFactorization)
    y = SNLU.solve!(cache.u, F, cache.b)
    # Static pivoting never aborts: a numerically singular system factors with
    # perturbed pivots and produces a finite but meaningless solution. When
    # pivots were perturbed (rare), verify the residual (one sparse mat-vec)
    # so singularity surfaces as `Infeasible` instead of a silent `Success`.
    ok = all(isfinite, y)
    if ok && SNLU.nperturbed(F) > 0
        r = F.A * y
        r .-= cache.b
        bn = LinearAlgebra.norm(cache.b)
        ok = LinearAlgebra.norm(r) <= 1.0e-6 * max(bn, floatmin(real(eltype(r))))
    end
    return if ok
        SciMLBase.build_linear_solution(
            alg, y, nothing, nothing; retcode = ReturnCode.Success
        )
    else
        @SciMLMessage(
            "Solver produced a non-finite or inaccurate solution; matrix is likely singular",
            cache.verbose, :solver_failure
        )
        SciMLBase.build_linear_solution(
            alg, cache.u, nothing, nothing; retcode = ReturnCode.Infeasible
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

# Generic element types (e.g. BigFloat): SparseColumnPivotedQR is pure-Julia and
# factors any `Number` element type. Returning a matching-eltype placeholder
# (rather than `nothing`) keeps the default polyalgorithm's
# `:SparseColumnPivotedQRFactorization` slot concretely typed so the sparse-LU
# singular fallback (`_do_sparse_qr_fallback`) can `setfield!` a real
# factorization into it for non-BLAS eltypes.
function LinearSolve.init_cacheval(
        alg::SparseColumnPivotedQRFactorization, A::AbstractSparseArray{T, <:Integer},
        b, u, Pl, Pr, maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    ) where {T <: Number}
    return SCPQR.scpqr(sparse(reshape([one(T)], 1, 1)))
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
    A = LinearSolve.reduce_operand!(cache.sparse_reduction, A)
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
    y = LinearSolve._ldiv!(cache.u, F, cache.b)
    return SciMLBase.build_linear_solution(
        alg, y, nothing, nothing; retcode = ReturnCode.Success
    )
end

# SparseColumnPivotedQR's ldiv! only accepts vector right-hand sides; batched
# (matrix) right-hand sides solve column-by-column against the one factorization.
function LinearSolve._ldiv!(
        x::AbstractMatrix,
        F::SCPQR.SparseColumnPivotedQRFactorization, b::AbstractMatrix
    )
    for j in axes(b, 2)
        ldiv!(view(x, :, j), F, view(b, :, j))
    end
    return x
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
                x::AbstractVecOrMat,
                A::SparseArrays.CHOLMOD.Factor, b::AbstractVecOrMat
            )
            x .= A \ b
        end
    end

    # ldiv!() for SPQR was added in 1.13: https://github.com/JuliaSparse/SparseArrays.jl/pull/676
    @static if VERSION < v"1.13"
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
    end

    # SPQR has no in-place matrix (batched) ldiv! on any current Julia version,
    # so route batched right-hand sides through the allocating `\`.
    function LinearSolve._ldiv!(
            x::AbstractMatrix,
            A::SparseArrays.SPQR.QRSparse, b::AbstractMatrix
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
# The `size(b, 1) <= 1_000` branch is the "small enough" fast path; the second is
# the "medium and sufficiently sparse" rule.
function LinearSolve.use_klulike_sparse_structure(A::AbstractSparseMatrixCSC, b)
    return size(b, 1) <= 1_000 ||
        (size(b, 1) <= 10_000 && length(nonzeros(A)) / length(A) < 2.0e-4)
end

function LinearSolve.defaultalg(
        A::AbstractSparseMatrixCSC{<:Union{Float64, ComplexF64}, Ti}, b,
        assump::OperatorAssumptions{Bool}
    ) where {Ti}
    klulike = LinearSolve.use_klulike_sparse_structure(A, b)
    return if assump.issq
        # Less structure → PureKLU; more structure → the pure-Julia supernodal
        # left-right-looking LU (both slots resolve to pure-Julia solvers, so
        # the sparse LU default no longer depends on Base.USE_GPL_LIBS).
        if klulike
            LinearSolve.DefaultLinearSolver(LinearSolve.DefaultAlgorithmChoice.KLUFactorization)
        else
            LinearSolve.DefaultLinearSolver(LinearSolve.DefaultAlgorithmChoice.UMFPACKFactorization)
        end
    else
        @static if Base.USE_GPL_LIBS
            # Sparse QR: less structure → SparseColumnPivotedQR (pure-Julia);
            # more structure → SuiteSparse SPQR (`QRFactorization`).
            if klulike
                LinearSolve.DefaultLinearSolver(LinearSolve.DefaultAlgorithmChoice.SparseColumnPivotedQRFactorization)
            else
                LinearSolve.DefaultLinearSolver(LinearSolve.DefaultAlgorithmChoice.QRFactorization)
            end
        else
            # No SuiteSparse SPQR available: pure-Julia sparse QR throughout.
            LinearSolve.DefaultLinearSolver(LinearSolve.DefaultAlgorithmChoice.SparseColumnPivotedQRFactorization)
        end
    end
end

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

# --- Nonstructural-zero reduction (shared helper for sparse solvers) ---
#
# Two strategies, chosen per `OperatorAssumptions(...; nonstructural_zeros)`:
#
#   * union caching (`cache_union = true`): drop only entries numerically zero in
#     every solve so far — the kept set is the running union of ever-nonzero
#     positions, which only grows, so `reduced` is a valid superset of the current
#     nonzeros and the inner symbolic factorization stays reusable. Best when the
#     zeros are persistent (stable positions).
#
#   * per-solve dropzeros (`cache_union = false`): drop *this* matrix's zeros each
#     solve and hand the inner solver a fresh pattern. No cross-solve symbolic
#     caching is assumed (the inner solver re-analyzes when the pattern changes).
#     Best when the zeros are inconsistent (wobble too much for a stable union).
#
# In `auto` mode the reduction starts in union caching and switches to per-solve
# dropzeros if the union bloats past `NONPERSISTENT_ZERO_FRACTION` (the zeros turned out
# not to be persistent). `active` / `cache_union` are runtime fields (not types),
# so `init_sparse_reduction` returns a concrete type either way — type-stable.
mutable struct SparseReduction{Tv, Ti}
    active::Bool
    cache_union::Bool             # true: union caching; false: per-solve dropzeros
    auto::Bool                    # may switch union -> per-solve on bloat
    nstart_zeros::Int             # # stored entries zero on the starting matrix
    colptr::Vector{Ti}            # full stored pattern (reference, for validation)
    rowval::Vector{Ti}
    mask::Vector{Bool}            # over full nnz positions: kept (ever-nonzero)?
    keep::Vector{Int}             # kept positions into the full nzval (CSC order)
    reduced::SparseMatrixCSC{Tv, Ti}
    nanalyze::Int
    nrefactor::Int
    # Tracks whether the most recent per-solve dropzeros changed the structure
    # of `reduced` (i.e. nnz changed). Set to `false` after the first per-solve
    # dropzeros call so the first call never triggers a false "changed" signal.
    # Used by the DefaultLinearSolver to decide whether to invalidate cached
    # symbolic factorizations (Sparspak, KLU) whose symbolic reuse is only valid
    # when the matrix structure is stable across calls.
    reduced_nnz::Int
    structure_changed::Bool
end

function _persistent_reduced(
        A::AbstractSparseMatrixCSC{Tv, Ti}, colptr::Vector{Ti}, rowval, mask,
        nzval::AbstractVector{Tv}
    ) where {Ti, Tv}
    _, k = size(A)
    keep = Int[]
    rcolptr = Vector{Ti}(undef, k + 1)
    rrowval = Ti[]
    rnzval = Tv[]
    @inbounds for j in 1:k
        rcolptr[j] = length(rrowval) + 1
        for p in colptr[j]:(colptr[j + 1] - 1)
            if mask[p]
                push!(keep, p)
                push!(rrowval, rowval[p])
                push!(rnzval, nzval[p])
            end
        end
    end
    rcolptr[k + 1] = length(rrowval) + 1
    reduced = deepcopy(LinearSolve.make_SparseMatrixCSC(A))
    copyto!(getcolptr(reduced), rcolptr)
    resize!(rowvals(reduced), length(rrowval))
    copyto!(rowvals(reduced), rrowval)
    resize!(nonzeros(reduced), length(rnzval))
    copyto!(nonzeros(reduced), rnzval)
    return keep, reduced
end

function LinearSolve.init_sparse_reduction(
        A::AbstractSparseMatrixCSC{Tv, Ti}, assumptions
    ) where {Tv, Ti}
    nsz = LinearSolve.__nonstructural_zeros(assumptions)
    NZ = LinearSolve.NonstructuralZeros
    nz = nonzeros(A)
    active = if nsz == NZ.Persistent || nsz == NZ.Present
        true
    elseif nsz == NZ.None
        false
    else  # Auto
        !isempty(nz) &&
            count(iszero, nz) / length(nz) >= LinearSolve.PERSISTENT_ZERO_FRACTION_THRESHOLD
    end
    auto = nsz == NZ.Auto
    cache_union = nsz != NZ.Present     # Present => per-solve dropzeros from the start
    colptr = copy(getcolptr(A))
    rowval = copy(rowvals(A))
    if active
        mask = Bool[!iszero(v) for v in nz]
        nstart_zeros = count(iszero, nz)
        keep, reduced = _persistent_reduced(A, colptr, rowval, mask, nz)
        return SparseReduction{Tv, Ti}(
            true, cache_union, auto, nstart_zeros, colptr, rowval, mask, keep, reduced,
            1, 0, nnz(reduced), false
        )
    else
        reduced = LinearSolve.make_SparseMatrixCSC(A)
        return SparseReduction{Tv, Ti}(
            false, cache_union, auto, 0, colptr, rowval, Bool[], Int[], reduced,
            0, 0, 0, false
        )
    end
end

function LinearSolve.reduce_operand!(red::SparseReduction, A)
    red.active || return A
    nz = nonzeros(A)
    # A change in the stored nnz breaks union caching: an explicit `Persistent`
    # assumption promised a constant pattern (error), while `Auto` drops the union
    # and falls back to per-solve dropzeros, handled like the non-union case below.
    if red.cache_union && length(nz) != length(red.mask)
        red.auto || throw(ArgumentError("nonstructural_zeros reduction requires a constant \
                             stored sparsity pattern across solves (stored nnz changed)"))
        red.cache_union = false
    end
    # per-solve dropzeros: drop this matrix's own zeros and let the inner solver
    # re-analyze when the pattern changes (no cross-solve union assumed).
    if !red.cache_union
        new_reduced = deepcopy(LinearSolve.make_SparseMatrixCSC(A))
        dropzeros!(new_reduced)
        new_nnz = nnz(new_reduced)
        red.structure_changed = new_nnz != red.reduced_nnz
        red.reduced_nnz = new_nnz
        red.reduced = new_reduced
        red.nrefactor += 1
        return red.reduced
    end
    grew = false
    @inbounds for i in eachindex(nz)
        if !red.mask[i] && !iszero(nz[i])
            red.mask[i] = true
            grew = true
        end
    end
    if grew
        red.keep, red.reduced = _persistent_reduced(
            A, red.colptr, red.rowval, red.mask, nz
        )
        red.nanalyze += 1
        # auto mode: if more than NONPERSISTENT_ZERO_FRACTION of the starting zeros
        # have activated, the zeros are not persistent — stop caching the union and
        # drop per solve instead (this matrix's own zeros only). `activated` is
        # `keep` minus the initial keep count (`length(mask) - nstart_zeros`).
        activated = length(red.keep) - (length(red.mask) - red.nstart_zeros)
        if red.auto && red.nstart_zeros > 0 &&
                activated > LinearSolve.NONPERSISTENT_ZERO_FRACTION * red.nstart_zeros
            red.cache_union = false
            red.reduced = deepcopy(LinearSolve.make_SparseMatrixCSC(A))
            dropzeros!(red.reduced)
            # First per-solve reduction after switching from union caching.
            # No "previous" per-solve result, so no structure change yet.
            red.reduced_nnz = nnz(red.reduced)
            red.structure_changed = false
        end
    else
        rnz = nonzeros(red.reduced)
        keep = red.keep
        @inbounds @simd for j in eachindex(keep)
            rnz[j] = nz[keep[j]]
        end
        red.nrefactor += 1
    end
    return red.reduced
end

end
