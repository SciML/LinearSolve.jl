module LinearSolveElementalExt

using LinearSolve, Elemental, LinearAlgebra
using LinearSolve: LinearCache, LinearVerbosity, OperatorAssumptions
using SciMLBase: SciMLBase, ReturnCode

const ElementalNumeric = Union{Float32, Float64, ComplexF32, ComplexF64}

function _elemental_eltype(A)
    T = eltype(A)
    return T <: ElementalNumeric ? T : (T <: Complex ? ComplexF64 : Float64)
end

function _to_elemental_matrix(A::Base.AbstractVecOrMat, ::Type{T}) where {T}
    return convert(Elemental.Matrix{T}, Matrix{T}(A))
end

# Always copy: lu!/qr!/lq!/cholesky! are in-place. Returning A directly would
# corrupt the user's matrix and cause reinit! + re-solve to factorize garbage.
function _to_elemental_matrix(A::Elemental.Matrix, ::Type{T}) where {T}
    B = Elemental.Matrix(T)
    Elemental.resize!(B, size(A, 1), size(A, 2))
    copyto!(B, A)
    return B
end

function _b_to_elemental(b::AbstractVector, ::Type{T}) where {T}
    return convert(Elemental.Matrix{T}, reshape(Vector{T}(b), length(b), 1))
end

function _b_to_elemental(b::Elemental.Matrix{T}, ::Type{T}) where {T}
    return b
end

function _b_to_elemental(b::Elemental.Matrix, ::Type{T}) where {T}
    return convert(Elemental.Matrix{T}, convert(Base.Matrix{T}, b))
end

function _elemental_factorize(alg::ElementalJL, A_el::Elemental.Matrix)
    if alg.method === :LU
        return LinearAlgebra.lu!(A_el)
    elseif alg.method === :QR
        return LinearAlgebra.qr!(A_el)
    elseif alg.method === :LQ
        return LinearAlgebra.lq!(A_el)
    elseif alg.method === :Cholesky
        # Must call cholesky!(A_el) directly, not cholesky!(Hermitian(A_el)).
        # The Hermitian path returns LinearAlgebra.Cholesky whose .factors is a
        # raw Elemental.Matrix with no \ defined, silently falling back to Julia's
        # generic LU on already-destroyed data. The direct path returns
        # CholeskyMatrix{T} which has \ → ElSolveAfterCholesky.
        return LinearAlgebra.cholesky!(A_el)
    else
        error(
            "Unknown method $(alg.method) for ElementalJL. " *
                "Valid choices are :LU (default), :QR, :LQ, or :Cholesky."
        )
    end
end

function LinearSolve.init_cacheval(
        alg::ElementalJL, A, b, u, Pl, Pr, maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return nothing
end

function SciMLBase.solve!(cache::LinearCache, alg::ElementalJL; kwargs...)
    A = cache.A
    T = _elemental_eltype(A)

    if cache.isfresh
        A_el = _to_elemental_matrix(A, T)
        cache.cacheval = _elemental_factorize(alg, A_el)
        cache.isfresh = false
    end

    fact = LinearSolve.@get_cacheval(cache, :ElementalJL)
    x_jl = convert(Base.Matrix{T}, fact \ _b_to_elemental(cache.b, T))
    copyto!(cache.u, vec(x_jl))

    return SciMLBase.build_linear_solution(alg, cache.u, nothing, cache; retcode = ReturnCode.Success)
end

end # module
