module LinearSolveSparseArraysExt

using LinearSolve, LinearAlgebra
using SparseArrays
using SparseArrays: AbstractSparseMatrixCSC, nonzeros, rowvals, getcolptr

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
        abstol, reltol, verbose::Bool, assumptions::OperatorAssumptions)
    nothing, nothing
end

function LinearSolve.init_cacheval(
        alg::QRFactorization, A::Symmetric{<:Number, <:SparseMatrixCSC}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Bool,
        assumptions::OperatorAssumptions)
    return nothing
end

function LinearSolve.handle_sparsematrixcsc_lu(A::AbstractSparseMatrixCSC)
    lu(SparseMatrixCSC(size(A)..., getcolptr(A), rowvals(A), nonzeros(A)),
        check = false)
end

function LinearSolve.defaultalg(
        A::Symmetric{<:Number, <:SparseMatrixCSC}, b, ::OperatorAssumptions{Bool})
    LinearSolve.DefaultLinearSolver(LinearSolve.DefaultAlgorithmChoice.CHOLMODFactorization)
end

function LinearSolve.defaultalg(A::AbstractSparseMatrixCSC{Tv, Ti}, b,
        assump::OperatorAssumptions{Bool}) where {Tv, Ti}
    if assump.issq
        DefaultLinearSolver(DefaultAlgorithmChoice.SparspakFactorization)
    else
        error("Generic number sparse factorization for non-square is not currently handled")
    end
end

function LinearSolve.init_cacheval(alg::GenericFactorization,
        A::Union{Hermitian{T, <:SparseMatrixCSC},
            Symmetric{T, <:SparseMatrixCSC}}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Bool,
        assumptions::OperatorAssumptions) where {T}
    newA = copy(convert(AbstractMatrix, A))
    LinearSolve.do_factorization(alg, newA, b, u)
end

const PREALLOCATED_UMFPACK = SparseArrays.UMFPACK.UmfpackLU(SparseMatrixCSC(0, 0, [1],
    Int[], Float64[]))

function LinearSolve.init_cacheval(
        alg::UMFPACKFactorization, A::SparseMatrixCSC{Float64, Int}, b, u,
        Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Bool, assumptions::OperatorAssumptions)
    PREALLOCATED_UMFPACK
end

function LinearSolve.init_cacheval(
        alg::UMFPACKFactorization, A::AbstractSparseArray, b, u, Pl, Pr,
        maxiters::Int, abstol,
        reltol,
        verbose::Bool, assumptions::OperatorAssumptions)
    A = convert(AbstractMatrix, A)
    zerobased = SparseArrays.getcolptr(A)[1] == 0
    return SparseArrays.UMFPACK.UmfpackLU(SparseMatrixCSC(size(A)..., getcolptr(A),
        rowvals(A), nonzeros(A)))
end

function SciMLBase.solve!(
        cache::LinearSolve.LinearCache, alg::UMFPACKFactorization; kwargs...)
    A = cache.A
    A = convert(AbstractMatrix, A)
    if cache.isfresh
        cacheval = LinearSolve.@get_cacheval(cache, :UMFPACKFactorization)
        if alg.reuse_symbolic
            # Caches the symbolic factorization: https://github.com/JuliaLang/julia/pull/33738
            if alg.check_pattern && pattern_changed(cacheval, A)
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
            fact = lu(SparseMatrixCSC(size(A)..., getcolptr(A), rowvals(A), nonzeros(A)),
                check = false)
        end
        cache.cacheval = fact
        cache.isfresh = false
    end

    F = LinearSolve.@get_cacheval(cache, :UMFPACKFactorization)
    if F.status == SparseArrays.UMFPACK.UMFPACK_OK
        y = ldiv!(cache.u, F, cache.b)
        SciMLBase.build_linear_solution(alg, y, nothing, cache)
    else
        SciMLBase.build_linear_solution(
            alg, cache.u, nothing, cache; retcode = ReturnCode.Infeasible)
    end
end

const PREALLOCATED_KLU = KLU.KLUFactorization(SparseMatrixCSC(0, 0, [1], Int[],
    Float64[]))

function LinearSolve.init_cacheval(
        alg::KLUFactorization, A::SparseMatrixCSC{Float64, Int}, b, u, Pl,
        Pr,
        maxiters::Int, abstol, reltol,
        verbose::Bool, assumptions::OperatorAssumptions)
    PREALLOCATED_KLU
end

function LinearSolve.init_cacheval(
        alg::KLUFactorization, A::AbstractSparseArray, b, u, Pl, Pr,
        maxiters::Int, abstol,
        reltol,
        verbose::Bool, assumptions::OperatorAssumptions)
    A = convert(AbstractMatrix, A)
    return KLU.KLUFactorization(SparseMatrixCSC(size(A)..., getcolptr(A), rowvals(A),
        nonzeros(A)))
end

# TODO: guard this against errors
function SciMLBase.solve!(cache::LinearSolve.LinearCache, alg::KLUFactorization; kwargs...)
    A = cache.A
    A = convert(AbstractMatrix, A)
    if cache.isfresh
        cacheval = LinearSolve.@get_cacheval(cache, :KLUFactorization)
        if alg.reuse_symbolic
            if alg.check_pattern && pattern_changed(cacheval, A)
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
        SciMLBase.build_linear_solution(alg, y, nothing, cache)
    else
        SciMLBase.build_linear_solution(
            alg, cache.u, nothing, cache; retcode = ReturnCode.Infeasible)
    end
end

const PREALLOCATED_CHOLMOD = cholesky(SparseMatrixCSC(0, 0, [1], Int[], Float64[]))

function LinearSolve.init_cacheval(alg::CHOLMODFactorization,
        A::Union{SparseMatrixCSC{T, Int}, Symmetric{T, SparseMatrixCSC{T, Int}}}, b, u,
        Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Bool, assumptions::OperatorAssumptions) where {T <:
                                                                Union{Float32, Float64}}
    PREALLOCATED_CHOLMOD
end

function LinearSolve.init_cacheval(alg::NormalCholeskyFactorization,
        A::Union{AbstractSparseArray, LinearSolve.GPUArraysCore.AnyGPUArray,
            Symmetric{<:Number, <:AbstractSparseArray}}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Bool,
        assumptions::OperatorAssumptions)
    LinearSolve.ArrayInterface.cholesky_instance(convert(AbstractMatrix, A))
end

# Specialize QR for the non-square case
# Missing ldiv! definitions: https://github.com/JuliaSparse/SparseArrays.jl/issues/242
function LinearSolve._ldiv!(x::Vector,
        A::Union{SparseArrays.QR, LinearAlgebra.QRCompactWY,
            SparseArrays.SPQR.QRSparse,
            SparseArrays.CHOLMOD.Factor}, b::Vector)
    x .= A \ b
end

function LinearSolve._ldiv!(x::AbstractVector,
        A::Union{SparseArrays.QR, LinearAlgebra.QRCompactWY,
            SparseArrays.SPQR.QRSparse,
            SparseArrays.CHOLMOD.Factor}, b::AbstractVector)
    x .= A \ b
end

# Ambiguity removal
function LinearSolve._ldiv!(::LinearSolve.SVector,
        A::Union{SparseArrays.CHOLMOD.Factor, LinearAlgebra.QR,
            LinearAlgebra.QRCompactWY, SparseArrays.SPQR.QRSparse},
        b::AbstractVector)
    (A \ b)
end
function LinearSolve._ldiv!(::LinearSolve.SVector,
        A::Union{SparseArrays.CHOLMOD.Factor, LinearAlgebra.QR,
            LinearAlgebra.QRCompactWY, SparseArrays.SPQR.QRSparse},
        b::LinearSolve.SVector)
    (A \ b)
end

function pattern_changed(fact, A::SparseArrays.SparseMatrixCSC)
    !(SparseArrays.decrement(SparseArrays.getcolptr(A)) ==
      fact.colptr && SparseArrays.decrement(SparseArrays.getrowval(A)) ==
      fact.rowval)
end

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

LinearSolve.PrecompileTools.@compile_workload begin
    A = sprand(4, 4, 0.3) + I
    b = rand(4)
    prob = LinearProblem(A, b)
    sol = solve(prob, KLUFactorization())
    sol = solve(prob, UMFPACKFactorization())
end

end
