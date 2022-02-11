_ldiv!(x, A, b) = ldiv!(x, A, b)
function _ldiv!(x::Vector, A::Factorization, b::Vector)
    # workaround https://github.com/JuliaLang/julia/issues/43507
    copyto!(x, b)
    ldiv!(A, x)
end


function SciMLBase.solve(cache::LinearCache, alg::AbstractFactorization; kwargs...)
    if cache.isfresh
        fact = do_factorization(alg, cache.A, cache.b, cache.u)
        cache = set_cacheval(cache, fact)
    end
    y = _ldiv!(cache.u, cache.cacheval, cache.b)
    SciMLBase.build_linear_solution(alg,y,nothing,cache)
end

# Bad fallback: will fail if `A` is just a stand-in
# This should instead just create the factorization type.
init_cacheval(alg::AbstractFactorization, A, b, u, Pl, Pr, maxiters, abstol, reltol, verbose) = do_factorization(alg, convert(AbstractMatrix,A), b, u)

## LU Factorizations

struct LUFactorization{P} <: AbstractFactorization
    pivot::P
end

struct GenericLUFactorization{P} <: AbstractFactorization
    pivot::P
end

function LUFactorization()
    pivot = @static if VERSION < v"1.7beta"
        Val(true)
    else
        RowMaximum()
    end
    LUFactorization(pivot)
end

function GenericLUFactorization()
    pivot = @static if VERSION < v"1.7beta"
        Val(true)
    else
        RowMaximum()
    end
    GenericLUFactorization(pivot)
end

function do_factorization(alg::LUFactorization, A, b, u)
    A = convert(AbstractMatrix,A)
    if A isa SparseMatrixCSC
        return lu(A)
    else
        fact = lu!(A, alg.pivot)
    end
    return fact
end

function do_factorization(alg::GenericLUFactorization, A, b, u)
    A = convert(AbstractMatrix,A)
    fact = LinearAlgebra.generic_lufact!(A, alg.pivot)
    return fact
end

init_cacheval(alg::Union{LUFactorization,GenericLUFactorization}, A, b, u, Pl, Pr, maxiters, abstol, reltol, verbose) = ArrayInterface.lu_instance(convert(AbstractMatrix,A))

# This could be a GenericFactorization perhaps?
Base.@kwdef struct UMFPACKFactorization <: AbstractFactorization
    reuse_symbolic::Bool = true
end

function init_cacheval(alg::UMFPACKFactorization, A, b, u, Pl, Pr, maxiters, abstol, reltol, verbose)
    A = convert(AbstractMatrix,A)
    zerobased = SparseArrays.getcolptr(A)[1] == 0
    res = SuiteSparse.UMFPACK.UmfpackLU(C_NULL, C_NULL, size(A, 1), size(A, 2),
                    zerobased ? copy(SparseArrays.getcolptr(A)) : SuiteSparse.decrement(SparseArrays.getcolptr(A)),
                    zerobased ? copy(rowvals(A)) : SuiteSparse.decrement(rowvals(A)),
                    copy(nonzeros(A)), 0)
    finalizer(SuiteSparse.UMFPACK.umfpack_free_symbolic, res)
    res
end

function do_factorization(::UMFPACKFactorization, A, b, u)
    A = convert(AbstractMatrix,A)
    if A isa SparseMatrixCSC
        return lu(A)
    else
        error("Sparse LU is not defined for $(typeof(A))")
    end
end

function SciMLBase.solve(cache::LinearCache, alg::UMFPACKFactorization)
    A = cache.A
    A = convert(AbstractMatrix,A)
    if cache.isfresh
        if cache.cacheval !== nothing && alg.reuse_symbolic
            # If we have a cacheval already, run umfpack_symbolic to ensure the symbolic factorization exists
            # This won't recompute if it does.
            SuiteSparse.UMFPACK.umfpack_symbolic!(cache.cacheval)
            fact = lu!(cache.cacheval, A)
        else
            fact = do_factorization(alg, A, cache.b, cache.u)
        end
        cache = set_cacheval(cache, fact)
    end

    y = ldiv!(cache.u, cache.cacheval, cache.b)
    SciMLBase.build_linear_solution(alg,y,nothing,cache)
end

Base.@kwdef struct KLUFactorization <: AbstractFactorization
    reuse_symbolic::Bool = true
end

function init_cacheval(alg::KLUFactorization, A, b, u, Pl, Pr, maxiters, abstol, reltol, verbose)
    return KLU.KLUFactorization(convert(AbstractMatrix,A)) # this takes care of the copy internally.
end

function do_factorization(::KLUFactorization, A, b, u)
    A = convert(AbstractMatrix,A)
    if A isa SparseMatrixCSC
        return klu(A)
    else
        error("KLU is not defined for $(typeof(A))")
    end
end

function SciMLBase.solve(cache::LinearCache, alg::KLUFactorization)
    A = cache.A
    A = convert(AbstractMatrix,A)
    if cache.isfresh
        if cache.cacheval !== nothing && alg.reuse_symbolic
            # If we have a cacheval already, run umfpack_symbolic to ensure the symbolic factorization exists
            # This won't recompute if it does.
            KLU.klu_analyze!(cache.cacheval)
            if cache.cacheval._numeric === C_NULL # We MUST have a numeric factorization for reuse, unlike UMFPACK.
                KLU.klu_factor!(cache.cacheval)
            end
            fact = KLU.klu!(cache.cacheval, A)
        else
            fact = do_factorization(alg, A, cache.b, cache.u)
        end
        cache = set_cacheval(cache, fact)
    end

    y = ldiv!(cache.u, cache.cacheval, cache.b)
    SciMLBase.build_linear_solution(alg,y,nothing,cache)
end

## QRFactorization

struct QRFactorization{P} <: AbstractFactorization
    pivot::P
    blocksize::Int
    inplace::Bool
end

function QRFactorization(inplace = true)
    pivot = @static if VERSION < v"1.7beta"
        Val(false)
    else
        NoPivot()
    end
    QRFactorization(pivot, 16, inplace)
end

function do_factorization(alg::QRFactorization, A, b, u)
    A = convert(AbstractMatrix,A)
    if alg.inplace
        fact = qr!(A, alg.pivot)
    else
        fact = qr(A) # CUDA.jl does not allow other args!
    end
    return fact
end

## SVDFactorization

struct SVDFactorization{A} <: AbstractFactorization
    full::Bool
    alg::A
end

SVDFactorization() = SVDFactorization(false, LinearAlgebra.DivideAndConquer())

function do_factorization(alg::SVDFactorization, A, b, u)
    A = convert(AbstractMatrix,A)
    fact = svd!(A; full = alg.full, alg = alg.alg)
    return fact
end

## GenericFactorization

struct GenericFactorization{F} <: AbstractFactorization
    fact_alg::F
end

GenericFactorization(;fact_alg = LinearAlgebra.factorize) =
    GenericFactorization(fact_alg)

function do_factorization(alg::GenericFactorization, A, b, u)
    A = convert(AbstractMatrix,A)
    fact = alg.fact_alg(A)
    return fact
end

init_cacheval(alg::GenericFactorization{typeof(lu)}, A, b, u, Pl, Pr, maxiters, abstol, reltol, verbose) = ArrayInterface.lu_instance(convert(AbstractMatrix,A))
init_cacheval(alg::GenericFactorization{typeof(lu!)}, A, b, u, Pl, Pr, maxiters, abstol, reltol, verbose) = ArrayInterface.lu_instance(convert(AbstractMatrix,A))

init_cacheval(alg::GenericFactorization{typeof(lu)}, A::StridedMatrix{<:LinearAlgebra.BlasFloat}, b, u, Pl, Pr, maxiters, abstol, reltol, verbose) = ArrayInterface.lu_instance(A)
init_cacheval(alg::GenericFactorization{typeof(lu!)}, A::StridedMatrix{<:LinearAlgebra.BlasFloat}, b, u, Pl, Pr, maxiters, abstol, reltol, verbose) = ArrayInterface.lu_instance(A)
init_cacheval(alg::GenericFactorization{typeof(lu)}, A::Diagonal, b, u, Pl, Pr, maxiters, abstol, reltol, verbose) = Diagonal(inv.(A.diag))
init_cacheval(alg::GenericFactorization{typeof(lu)}, A::Tridiagonal, b, u, Pl, Pr, maxiters, abstol, reltol, verbose) = ArrayInterface.lu_instance(A)
init_cacheval(alg::GenericFactorization{typeof(lu!)}, A::Diagonal, b, u, Pl, Pr, maxiters, abstol, reltol, verbose) = Diagonal(inv.(A.diag))
init_cacheval(alg::GenericFactorization{typeof(lu!)}, A::Tridiagonal, b, u, Pl, Pr, maxiters, abstol, reltol, verbose) = ArrayInterface.lu_instance(A)

init_cacheval(alg::GenericFactorization, A::Diagonal, b, u, Pl, Pr, maxiters, abstol, reltol, verbose) = Diagonal(inv.(A.diag))
init_cacheval(alg::GenericFactorization, A::Tridiagonal, b, u, Pl, Pr, maxiters, abstol, reltol, verbose) = ArrayInterface.lu_instance(A)
init_cacheval(alg::GenericFactorization, A::SymTridiagonal{T,V}, b, u, Pl, Pr, maxiters, abstol, reltol, verbose) where {T,V} = LinearAlgebra.LDLt{T,SymTridiagonal{T,V}}(A)

function init_cacheval(alg::Union{GenericFactorization,GenericFactorization{typeof(bunchkaufman!)},GenericFactorization{typeof(bunchkaufman)}},
                       A::Union{Hermitian,Symmetric}, b, u, Pl, Pr, maxiters, abstol, reltol, verbose)
    BunchKaufman(A.data, Array(1:size(A,1)), A.uplo, true, false, 0)
end

function init_cacheval(alg::Union{GenericFactorization{typeof(bunchkaufman!)},GenericFactorization{typeof(bunchkaufman)}},
                       A::StridedMatrix{<:LinearAlgebra.BlasFloat}, b, u, Pl, Pr, maxiters, abstol, reltol, verbose)
   if eltype(A) <: Complex
       return bunchkaufman!(Hermitian(A))
   else
       return bunchkaufman!(Symmetric(A))
   end
end

# Fallback, tries to make nonsingular and just factorizes
# Try to never use it.
function init_cacheval(alg::Union{QRFactorization,SVDFactorization,GenericFactorization}, A, b, u, Pl, Pr, maxiters, abstol, reltol, verbose)
    newA = copy(convert(AbstractMatrix,A))
    if newA isa SparseMatrixCSC
        fill!(nonzeros(newA),true)
    else
        fill!(newA,true)
    end
    do_factorization(alg, newA, b, u)
end

# Ambiguity handling dispatch
function init_cacheval(alg::Union{QRFactorization,SVDFactorization}, A::StridedMatrix{<:LinearAlgebra.BlasFloat}, b, u, Pl, Pr, maxiters, abstol, reltol, verbose)
    newA = copy(convert(AbstractMatrix,A))
    if newA isa SparseMatrixCSC
        fill!(nonzeros(newA),true)
    else
        fill!(newA,true)
    end
    do_factorization(alg, newA, b, u)
end

# Cholesky needs the posdef matrix, for GenericFactorization assume structure is needed
function init_cacheval(alg::Union{GenericFactorization,GenericFactorization{typeof(cholesky)},GenericFactorization{typeof(cholesky!)}}, A, b, u, Pl, Pr, maxiters, abstol, reltol, verbose)
    newA = copy(convert(AbstractMatrix,A))
    do_factorization(alg, newA, b, u)
end

# Ambiguity handling dispatch
function init_cacheval(alg::Union{GenericFactorization,GenericFactorization{typeof(cholesky)},GenericFactorization{typeof(cholesky!)}}, A::StridedMatrix{<:LinearAlgebra.BlasFloat}, b, u, Pl, Pr, maxiters, abstol, reltol, verbose)
    newA = copy(convert(AbstractMatrix,A))
    do_factorization(alg, newA, b, u)
end

## RFLUFactorization

struct RFWrapper{P,T}
    RFWrapper(::Val{P},::Val{T}) where {P,T} = new{P,T}()
end
(::RFWrapper{P,T})(A) where {P,T} = RecursiveFactorization.lu!(A,Val(P),Val(T))

RFLUFactorization(;pivot = Val(true), thread = Val(true)) = GenericFactorization(;fact_alg=RFWrapper(pivot,thread))

init_cacheval(alg::GenericFactorization{<:RFWrapper}, A, b, u, Pl, Pr, maxiters, abstol, reltol, verbose) = ArrayInterface.lu_instance(convert(AbstractMatrix,A))
init_cacheval(alg::GenericFactorization{<:RFWrapper}, A::StridedMatrix{<:LinearAlgebra.BlasFloat}, b, u, Pl, Pr, maxiters, abstol, reltol, verbose) = ArrayInterface.lu_instance(convert(AbstractMatrix,A))
