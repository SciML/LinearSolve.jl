module LinearSolveSparseArraysExt

using LinearSolve
import LinearSolve: SciMLBase, LinearAlgebra, PrecompileTools, init_cacheval
using LinearSolve: DefaultLinearSolver, DefaultAlgorithmChoice
using SparseArrays
using SparseArrays: AbstractSparseMatrixCSC, nonzeros, rowvals, getcolptr

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
function LinearSolve._ldiv!(::SVector,
    A::Union{SparseArrays.CHOLMOD.Factor, LinearAlgebra.QR,
        LinearAlgebra.QRCompactWY, SparseArrays.SPQR.QRSparse},
    b::AbstractVector)
(A \ b)
end
function LinearSolve._ldiv!(::SVector,
    A::Union{SparseArrays.CHOLMOD.Factor, LinearAlgebra.QR,
        LinearAlgebra.QRCompactWY, SparseArrays.SPQR.QRSparse},
    b::SVector)
(A \ b)
end

function LinearSolve.pattern_changed(fact, A::SparseArrays.SparseMatrixCSC)
    !(SparseArrays.decrement(SparseArrays.getcolptr(A)) ==
    fact.colptr && SparseArrays.decrement(SparseArrays.getrowval(A)) ==
    fact.rowval)
end

const PREALLOCATED_UMFPACK = SparseArrays.UMFPACK.UmfpackLU(SparseMatrixCSC(0, 0, [1],
    Int[], Float64[]))

function init_cacheval(alg::UMFPACKFactorization,
        A, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Bool, assumptions::OperatorAssumptions)
    nothing
end

function init_cacheval(alg::UMFPACKFactorization, A::SparseMatrixCSC{Float64, Int}, b, u,
        Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Bool, assumptions::OperatorAssumptions)
    PREALLOCATED_UMFPACK
end

function init_cacheval(alg::UMFPACKFactorization, A::AbstractSparseArray, b, u, Pl, Pr,
        maxiters::Int, abstol,
        reltol,
        verbose::Bool, assumptions::OperatorAssumptions)
    A = convert(AbstractMatrix, A)
    zerobased = SparseArrays.getcolptr(A)[1] == 0
    return SparseArrays.UMFPACK.UmfpackLU(SparseMatrixCSC(size(A)..., getcolptr(A),
        rowvals(A), nonzeros(A)))
end

function SciMLBase.solve!(cache::LinearCache, alg::UMFPACKFactorization; kwargs...)
    A = cache.A
    A = convert(AbstractMatrix, A)
    if cache.isfresh
        cacheval = @get_cacheval(cache, :UMFPACKFactorization)
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

    F = @get_cacheval(cache, :UMFPACKFactorization)
    if F.status == SparseArrays.UMFPACK.UMFPACK_OK
        y = ldiv!(cache.u, F, cache.b)
        SciMLBase.build_linear_solution(alg, y, nothing, cache)
    else
        SciMLBase.build_linear_solution(
            alg, cache.u, nothing, cache; retcode = ReturnCode.Infeasible)
    end
end

const PREALLOCATED_CHOLMOD = cholesky(SparseMatrixCSC(0, 0, [1], Int[], Float64[]))

function init_cacheval(alg::CHOLMODFactorization,
        A, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Bool, assumptions::OperatorAssumptions)
    nothing
end

function init_cacheval(alg::CHOLMODFactorization,
        A::Union{SparseMatrixCSC{T, Int}, Symmetric{T, SparseMatrixCSC{T, Int}}}, b, u,
        Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Bool, assumptions::OperatorAssumptions) where {T <:
                                                                Union{Float32, Float64}}
    PREALLOCATED_CHOLMOD
end

function SciMLBase.solve!(cache::LinearCache, alg::CHOLMODFactorization; kwargs...)
    A = cache.A
    A = convert(AbstractMatrix, A)

    if cache.isfresh
        cacheval = @get_cacheval(cache, :CHOLMODFactorization)
        fact = cholesky(A; check = false)
        if !LinearAlgebra.issuccess(fact)
            ldlt!(fact, A; check = false)
        end
        cache.cacheval = fact
        cache.isfresh = false
    end

    cache.u .= @get_cacheval(cache, :CHOLMODFactorization) \ cache.b
    SciMLBase.build_linear_solution(alg, cache.u, nothing, cache)
end

function LinearSolve.defaultalg(
        A::Symmetric{<:Number, <:SparseMatrixCSC}, b, ::OperatorAssumptions{Bool})
    DefaultLinearSolver(DefaultAlgorithmChoice.CHOLMODFactorization)
end

function LinearSolve.defaultalg(A::AbstractSparseMatrixCSC{Tv, Ti}, b,
        assump::OperatorAssumptions{Bool}) where {Tv, Ti}
    if assump.issq
        DefaultLinearSolver(DefaultAlgorithmChoice.SparspakFactorization)
    else
        error("Generic number sparse factorization for non-square is not currently handled")
    end
end

function LinearSolve.defaultalg(A::AbstractSparseMatrixCSC{<:Union{Float64, ComplexF64}, Ti}, b,
        assump::OperatorAssumptions{Bool}) where {Ti}
    if assump.issq
        if length(b) <= 10_000 && length(nonzeros(A)) / length(A) < 2e-4
            DefaultLinearSolver(DefaultAlgorithmChoice.KLUFactorization)
        else
            DefaultLinearSolver(DefaultAlgorithmChoice.UMFPACKFactorization)
        end
    else
        DefaultLinearSolver(DefaultAlgorithmChoice.QRFactorization)
    end
end

PrecompileTools.@compile_workload begin
    A = sprand(4, 4, 0.3) + I
    b = rand(4)
    prob = LinearProblem(A, b)
    sol = solve(prob, KLUFactorization())
    sol = solve(prob, UMFPACKFactorization())
end

end
