## IterativeSolvers.jl

mutable struct LinSolveIterativeSolvers{F,PL,PR,AR,A}
    generate_iterator::F
    iterable::Any
    Pl::PL
    Pr::PR
    args::AR
    kwargs::A
end

LinSolveIterativeSolvers(
    generate_iterator,
    args...;
    Pl = IterativeSolvers.Identity(),
    Pr = IterativeSolvers.Identity(),
    kwargs...,
) = LinSolveIterativeSolvers(generate_iterator, nothing, Pl, Pr, args, kwargs)

function (f::LinSolveIterativeSolvers)(
    x,
    A,
    b,
    update_matrix = false;
    Pl = nothing,
    Pr = nothing,
    reltol = eps(eltype(x)),
    kwargs...,
)
    if f.iterable === nothing
        Pl = ComposePreconditioner(f.Pl, Pl, true)
        Pr = ComposePreconditioner(f.Pr, Pr, false)

        reltol = checkreltol(reltol)
        f.iterable = f.generate_iterator(
            x,
            A,
            b,
            f.args...;
            initially_zero = true,
            restart = 5,
            maxiter = 5,
            abstol = 1e-16,
            reltol = reltol,
            Pl = Pl,
            Pr = Pr,
            kwargs...,
            f.kwargs...,
        )
    end
    x .= false
    iter = f.iterable
    purge_history!(iter, x, b)

    for residual in iter
    end

    return nothing
end

function (p::LinSolveIterativeSolvers)(::Type{Val{:init}}, f, u0_prototype)
    LinSolveIterativeSolvers(
        p.generate_iterator,
        nothing,
        p.Pl,
        p.Pr,
        p.args,
        p.kwargs,
    )
end

# scaling for iterative solvers
struct ScaleVector{T}
    x::T
    isleft::Bool
end

function LinearAlgebra.ldiv!(v::ScaleVector, x)
    vx = vec(v.x)
    if v.isleft
        return @.. x = x * vx
    else
        return @.. x = x / vx
    end
end

function LinearAlgebra.ldiv!(y, v::ScaleVector, x)
    vx = vec(v.x)
    if v.isleft
        return @.. y = x * vx
    else
        return @.. y = x / vx
    end
end

struct ComposePreconditioner{P,S<:Union{ScaleVector,Nothing}}
    P::P
    scale::S
    isleft::Bool
end

function LinearAlgebra.ldiv!(v::ComposePreconditioner, x)
    isid = v.P isa IterativeSolvers.Identity
    isid || ldiv!(v.P, x)
    s = v.scale
    s === nothing && return x
    ldiv!(s, x)
    return x
end

function LinearAlgebra.ldiv!(y, v::ComposePreconditioner, x)
    isid = v.P isa IterativeSolvers.Identity
    isid || ldiv!(y, v.P, x)
    s = v.scale
    s === nothing && return x
    if isid
        ldiv!(y, s, x)
    else
        if v.isleft
            @.. y = y * s.x
        else
            @.. y = y / s.x
        end
    end
    return y
end

function purge_history!(iter::IterativeSolvers.GMRESIterable, x, b)
    iter.k = 1
    iter.x = x
    iter.b = b

    iter.residual.current = IterativeSolvers.init!(
        iter.arnoldi,
        iter.x,
        iter.b,
        iter.Pl,
        iter.Ax,
        initially_zero = true,
    )
    IterativeSolvers.init_residual!(iter.residual, iter.residual.current)
    iter.β = iter.residual.current
    nothing
end

function Base.resize!(f::LinSolveIterativeSolvers, i)
    f.iterable = nothing
end


## Krylov.jl
struct LinSolveKrylov{S,A,K}
    solver::S
    args::A
    kwargs::K
end

LinSolveKrylov(solver = Krylov.gmres, args...; kwargs...) =
    LinSolveKrylov(solver, args, _krylov_update_kwargs(kwargs))

function (l::LinSolveKrylov)(x, A, b, matrix_updated = false)
    x .= l.solver(A, b, l.args...; l.kwargs...)[1]
    return x
end

(l::LinSolveKrylov)(::Type{Val{:init}}, f, u0_prototype) = l


# Make the kwargs consistent
function _krylov_update_kwargs(kwargs::Base.Pairs)
    key_list = []
    value_list = []
    for (k, v) in kwargs
        if k == :reltol
            push!(key_list, :rtol)
        elseif k == :abstol
            push!(key_list, :atol)
        elseif k ∈ [:maxiter, :maxiters]
            push!(key_list, :itmax)
        else
            push!(key_list, k)
        end
        push!(value_list, v)
    end
    return pairs(NamedTuple(zip(key_list, value_list)))
end


## Use Krylov if CUDA is loaded to be safe else just return IterativeSolvers
LinSolveGMRES(args...; kwargs...) =
    !is_cuda_available() ?
    LinSolveIterativeSolvers(
        IterativeSolvers.gmres_iterable!,
        args...;
        kwargs...,
    ) : LinSolveKrylov(Krylov.gmres, args...; kwargs...)

LinSolveCG(args...; kwargs...) =
    !is_cuda_available() ?
    LinSolveIterativeSolvers(
        IterativeSolvers.cg_iterator!,
        args...;
        kwargs...,
    ) : LinSolveKrylov(Krylov.cg, args...; kwargs...)

LinSolveBiCGStabl(args...; kwargs...) =
    !is_cuda_available() ?
    LinSolveIterativeSolvers(
        IterativeSolvers.bicgstabl_iterator!,
        args...;
        kwargs...,
    ) : LinSolveKrylov(Krylov.bicgstab, args...; kwargs...)

LinSolveChebyshev(args...; kwargs...) = LinSolveIterativeSolvers(
    IterativeSolvers.chebyshev_iterable!,
    args...;
    kwargs...,
)

LinSolveMINRES(args...; kwargs...) =
    !is_cuda_available() ?
    LinSolveIterativeSolvers(
        IterativeSolvers.minres_iterable!,
        args...;
        kwargs...,
    ) : LinSolveKrylov(Krylov.minres, args...; kwargs...)
