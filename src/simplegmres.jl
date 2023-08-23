"""
    SimpleGMRES(; restart::Int = 20, blocksize::Int = 0)

A simple GMRES implementation for square non-Hermitian linear systems.

This implementation handles Block Diagonal Matrices with Uniformly Sized Square Blocks with
specialized dispatches.

## Arguments

* `restart::Int = 20`: the number of iterations before restarting. Must be a strictly
  positive integer.
* `blocksize::Int = 0`: If blocksize is `> 0`, the solver assumes that the matrix has a
  uniformly sized block diagonal structure with square blocks of size `blocksize`. Misusing
  this option will lead to incorrect results.
    * If this is set `≤ 0` and during runtime we get a Block Diagonal Matrix, then we will
      check if the specialized dispatch can be used.

!!! warning

    Most users should be using the `KrylovJL_GMRES` solver instead of this implementation.

!!! tip

    We can automatically detect if the matrix is a Block Diagonal Matrix with Uniformly
    Sized Square Blocks. If this is the case, then we can use a specialized dispatch.
    However, on most modern systems performing a single matrix-vector multiplication is
    faster than performing multiple smaller matrix-vector multiplications (as in the case
    of Block Diagonal Matrix). We recommend making the matrix dense (if size permits) and
    specifying the `blocksize` argument.
"""
struct SimpleGMRES{UBD} <: AbstractKrylovSubspaceMethod
    restart::Int
    blocksize::Int

    function SimpleGMRES(; restart::Int = 20, blocksize::Int = 0)
        @assert restart≥1 "restart must be greater than or equal to 1"
        return new{blocksize > 0}(restart, blocksize)
    end
end

struct SimpleGMRESCache{UBD, T, QType, HType, xType, rType, βe₁Type, AType, bType, βType}
    M::Int
    N::Int
    maxiters::Int
    blocksize::Int
    ϵ::T
    Q::QType
    H::HType
    x::xType
    r::rType
    βe₁::βe₁Type
    A::AType
    b::bType
    β::βType
    abstol::T

    function SimpleGMRESCache{UBD}(M, N, maxiters, blocksize, ϵ, Q, H, x, r, βe₁, A, b, β,
        abstol) where {UBD}
        return new{UBD, typeof(ϵ), typeof(Q), typeof(H), typeof(x), typeof(r), typeof(βe₁),
            typeof(A), typeof(b), typeof(β)}(M, N, maxiters, blocksize, ϵ, Q, H, x, r, βe₁,
            A, b, β, abstol)
    end
end

_no_preconditioner(::Nothing) = true
_no_preconditioner(::IdentityOperator) = true
_no_preconditioner(::UniformScaling) = true
_no_preconditioner(_) = false

_norm2(x) = norm(x, 2)
_norm2(x, dims) = .√(sum(abs2, x; dims))

function init_cacheval(alg::SimpleGMRES{UDB}, args...; kwargs...) where {UDB}
    return _init_cacheval(Val(UDB), alg, args...; kwargs...)
end

function _init_cacheval(::Val{false}, alg::SimpleGMRES, A, b, u, Pl, Pr, maxiters::Int,
    abstol, ::Any, ::Bool, ::OperatorAssumptions; zeroinit = true, kwargs...)
    if zeroinit
        return SimpleGMRESCache{false}(0, 0, maxiters, alg.blocksize, zero(eltype(u)),
            similar(b, 0, 0), similar(b, 0, 0), u, similar(b, 0), similar(b, 0),
            A, b, zero(eltype(u)), abstol)
    end

    @assert _no_preconditioner(Pl)&&_no_preconditioner(Pr) "Preconditioning not supported! Use KrylovJL_GMRES instead."
    N = LinearAlgebra.checksquare(A)
    @assert N == length(b) "The size of `A` and `b` must match."
    T = eltype(u)
    M = min(maxiters, alg.restart)
    ϵ = eps(T)

    # Initialize the Cache
    ## Use `b` since `A` might be an operator
    Q = similar(b, length(b), M + 1)
    H = similar(b, M + 1, M)
    fill!(H, zero(T))

    mul!(@view(Q[:, 1]), A, u, T(-1), T(0))  # r0 <- A u
    axpy!(T(1), b, @view(Q[:, 1]))  # r0 <- r0 - b
    β = _norm2(@view(Q[:, 1]))
    Q[:, 1] ./= β

    x = u
    r = similar(b)
    βe₁ = similar(b, M + 1)
    fill!(βe₁, 0)
    βe₁[1:1] .= β  # Avoid the scalar indexing error

    return SimpleGMRESCache{false}(M, N, maxiters, alg.blocksize, ϵ, Q, H, x, r, βe₁, A, b,
        β, abstol)
end

function _init_cacheval(::Val{true}, alg::SimpleGMRES, A, b, u, Pl, Pr, maxiters::Int,
    abstol, ::Any, ::Bool, ::OperatorAssumptions; zeroinit = true,
    blocksize = alg.blocksize)
    if zeroinit
        return SimpleGMRESCache{true}(0, 0, maxiters, alg.blocksize, zero(eltype(u)),
            similar(b, 0, 0, 0), similar(b, 0, 0, 0), u, similar(b, 0), similar(b, 0, 0),
            A, b, similar(b, 0, 0), abstol)
    end

    @assert _no_preconditioner(Pl)&&_no_preconditioner(Pr) "Preconditioning not supported! Use KrylovJL_GMRES instead."
    N = LinearAlgebra.checksquare(A)
    @assert mod(N, blocksize)==0 "The blocksize must divide the size of the matrix."
    @assert N==length(b) "The size of `A` and `b` must match."
    T = eltype(u)
    M = min(maxiters, alg.restart)
    ϵ = eps(T)
    bsize = N ÷ blocksize

    # Initialize the Cache
    ## Use `b` since `A` might be an operator
    Q = similar(b, blocksize, M + 1, bsize)
    H = similar(b, M + 1, M, bsize)
    fill!(H, zero(T))

    mul!(vec(@view(Q[:, 1, :])), A, u, T(-1), T(0))  # r0 <- A u
    axpy!(T(1), b, vec(@view(Q[:, 1, :])))  # r0 <- r0 - b
    β = _norm2(@view(Q[:, 1, :]), 1)
    Q[:, 1, :] ./= β

    x = u
    r = similar(b)
    βe₁ = similar(b, M + 1, bsize)
    fill!(βe₁, 0)
    βe₁[1, :] .= vec(β)  # Avoid the scalar indexing error

    return SimpleGMRESCache{true}(M, N, maxiters, blocksize, ϵ, Q, H, x, r, βe₁, A, b,
        β, abstol)
end

default_alias_A(::SimpleGMRES, ::Any, ::Any) = false
default_alias_b(::SimpleGMRES, ::Any, ::Any) = false

function SciMLBase.solve!(cache::LinearCache, alg::SimpleGMRES; kwargs...)
    if cache.isfresh
        solver = init_cacheval(alg, cache.A, cache.b, cache.u, cache.Pl, cache.Pr,
            cache.maxiters, cache.abstol, cache.reltol, cache.verbose,
            cache.assumptions; zeroinit = false)
        cache.cacheval = solver
        cache.isfresh = false
    end
    return SciMLBase.solve!(cache.cacheval, cache)
end

function SciMLBase.solve!(cache::SimpleGMRESCache{false, T},
    lincache::LinearCache) where {T}
    @unpack M, N, maxiters, ϵ, Q, H, x, r, βe₁, A, b, β, abstol = cache
    res_norm = β

    # FIXME: The performance for this is quite bad when compared to the KrylovJL_GMRES
    #        version
    for _ in 1:((maxiters ÷ M) + 1)
        for j in 1:M
            Qⱼ₊₁ = @view(Q[:, j + 1])
            mul!(Qⱼ₊₁, A, @view(Q[:, j]))  # Q(:,j+1) <- A Q(:, j)
            for i in 1:j
                H[i, j] = dot(@view(Q[:, i]), Qⱼ₊₁)
                axpy!(-H[i, j], @view(Q[:, i]), Qⱼ₊₁)
            end
            H[j + 1, j] = _norm2(Qⱼ₊₁)
            H[j + 1, j] > ϵ && (Qⱼ₊₁ ./= H[j + 1, j])

            # FIXME: Figure out a way to avoid the allocation
            # Using views doesn't work very well with LinearSolve
            y = @view(H[1:(j + 1), 1:j]) \ @view(βe₁[1:(j + 1)])

            # Update the solution
            mul!(x, @view(Q[:, 1:j]), y)
            mul!(r, A, x, T(-1), T(0))
            axpy!(T(1), b, r)
            res_norm = _norm2(r)

            if res_norm < abstol
                return SciMLBase.build_linear_solution(lincache.alg, x, r, lincache;
                    retcode = ReturnCode.Success)
            end
        end

        # Restart
        Q[:, 1] = r ./ res_norm
        fill!(H, zero(T))
    end

    return SciMLBase.build_linear_solution(lincache.alg, x, r, lincache;
        retcode = ReturnCode.MaxIters)
end
