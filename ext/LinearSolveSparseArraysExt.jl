module LinearSolveSparseArraysExt

using LinearSolve: LinearSolve, BLASELTYPES, pattern_changed, ArrayInterface,
    CHOLMODFactorization, GenericFactorization,
    GenericLUFactorization,
    KLUFactorization, PureKLUFactorization, LUFactorization,
    NormalCholeskyFactorization,
    OperatorAssumptions, LinearVerbosity,
    QRFactorization, RFLUFactorization, UMFPACKFactorization,
    SparseColumnPivotedQRFactorization, solve
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

# --- SparseColumnPivotedQR: pure-Julia rank-revealing column-pivoted sparse QR ---
# The default sparse QR (non-square sparse systems) and the singular-LU fallback.

# SparseColumnPivotedQR is now CSC-native, so the matrix is passed straight
# through as a `SparseMatrixCSC` with no transpose/CSR round-trip.
_scpqr_csc(A) = SparseMatrixCSC(A)

# Element-typed preallocated factorizations. `CSRQRFactorization{T, real(T)}` is
# not parameterized on the index type, so one prealloc per element type suffices
# to fix the default cacheval slot's type (used by the singular-LU fallback).
const PREALLOCATED_SCPQR_F64 = SCPQR.csr_qr(_scpqr_csc(sparse(reshape([1.0], 1, 1))))
const PREALLOCATED_SCPQR_C64 = SCPQR.csr_qr(_scpqr_csc(sparse(reshape([ComplexF64(1)], 1, 1))))

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
        Acsc = _scpqr_csc(A)
        # Reuse the cached symbolic + numeric workspace via `csr_refactor!` when the
        # cached factorization has the same shape (it re-analyzes internally if the
        # pattern changed); otherwise build fresh. A size mismatch also rules out the
        # tiny preallocated factorization, forcing the first real factorization.
        fact = if alg.reuse_symbolic && cacheval isa SCPQR.CSRQRFactorization &&
                size(cacheval) == size(A)
            SCPQR.csr_refactor!(cacheval, Acsc)
        else
            SCPQR.csr_qr(Acsc; ordering = alg.ordering)
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
    return SCPQR.csr_qr(_scpqr_csc(convert(AbstractMatrix, A)))
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

end
