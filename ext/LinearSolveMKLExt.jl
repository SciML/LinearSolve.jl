module LinearSolveMKLExt

using MKL_jll
using LinearAlgebra: BlasInt, LU
using LinearAlgebra.LAPACK: require_one_based_indexing, chkfinite, chkstride1, chkargsok
using LinearAlgebra
const usemkl = MKL_jll.is_available()

using LinearSolve
using LinearSolve: ArrayInterface, MKLLUFactorization, @get_cacheval, LinearCache, SciMLBase

function getrf!(A::AbstractMatrix{<:Float64}; ipiv = similar(A, BlasInt, min(size(A,1),size(A,2))), info = Ref{BlasInt}(), check = false)
    require_one_based_indexing(A)
    check && chkfinite(A)
    chkstride1(A)
    m, n = size(A)
    lda  = max(1,stride(A, 2))

    if isempty(ipiv)
        ipiv = similar(A, BlasInt, min(size(A,1),size(A,2)))
    end

    ccall((:dgetrf_, MKL_jll.libmkl_rt), Cvoid,
            (Ref{BlasInt}, Ref{BlasInt}, Ptr{Float64},
            Ref{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}),
            m, n, A, lda, ipiv, info)
    chkargsok(info[])
    A, ipiv, info[] #Error code is stored in LU factorization type
end

default_alias_A(::MKLLUFactorization, ::Any, ::Any) = false
default_alias_b(::MKLLUFactorization, ::Any, ::Any) = false

function LinearSolve.init_cacheval(alg::MKLLUFactorization, A, b, u, Pl, Pr,
    maxiters::Int, abstol, reltol, verbose::Bool,
    assumptions::OperatorAssumptions)
    ArrayInterface.lu_instance(convert(AbstractMatrix, A))
end

function SciMLBase.solve!(cache::LinearCache, alg::MKLLUFactorization;
    kwargs...)
    A = cache.A
    A = convert(AbstractMatrix, A)
    if cache.isfresh
        cacheval = @get_cacheval(cache, :MKLLUFactorization)
        fact = LU(getrf!(A)...)
        cache.cacheval = fact
        cache.isfresh = false
    end
    y = ldiv!(cache.u, @get_cacheval(cache, :MKLLUFactorization), cache.b)
    SciMLBase.build_linear_solution(alg, y, nothing, cache)
end

end