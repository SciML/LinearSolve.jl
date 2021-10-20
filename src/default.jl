### Default Linsolve

# Try to be as smart as possible
# lu! if Matrix
# lu if sparse
# gmres if operator

mutable struct DefaultLinSolve
    A::Any
    iterable::Any
    openblas::Union{Bool,Nothing}
end

DefaultLinSolve() = DefaultLinSolve(nothing, nothing, nothing)

@noinline function checkreltol(reltol)
    if !(reltol isa Real)
        error(
            "Non real valued reltol is not supported by the linear iterative solvers. To customize tolerances for the linear iterative solvers, use the syntax like `KenCarp3(linsolve=LinSolveGMRES(abstol=1e-16,reltol=1e-16))`.",
        )
    end
    return reltol
end

function (p::DefaultLinSolve)(
    x,
    A::Union{AbstractMatrix,AbstractDiffEqOperator},
    b,
    update_matrix = false;
    reltol = nothing,
    kwargs...,
)
    if p.iterable isa Vector && eltype(p.iterable) <: LinearAlgebra.BlasInt # `iterable` here is the pivoting vector
        F = LU{eltype(A)}(A, p.iterable, zero(LinearAlgebra.BlasInt))
        ldiv!(x, F, b)
        return nothing
    end

    if update_matrix
        if A isa Matrix
            # if the user doesn't use OpenBLAS, we assume that is a better BLAS
            # implementation like MKL
            #
            # RecursiveFactorization seems to be consistantly winning below 100
            # https://discourse.julialang.org/t/ann-recursivefactorization-jl/39213
            if ArrayInterface.can_setindex(x) && p.openblas === nothing
                # cache it because it's more expensive now
                p.openblas = isopenblas()
            end

            if ArrayInterface.can_setindex(x) &&
               (size(A, 1) <= 100 || (p.openblas && size(A, 1) <= 500))
                p.A = RecursiveFactorization.lu!(A)
            else
                p.A = lu!(A)
            end
        elseif A isa Union{Tridiagonal,ForwardSensitivityJacobian}
            p.A = lu!(A)
        elseif A isa Union{SymTridiagonal}
            p.A = ldlt!(A)
        elseif A isa Union{Symmetric,Hermitian}
            p.A = bunchkaufman!(A)
        elseif A isa SparseMatrixCSC
            p.A = lu(A)
        elseif ArrayInterface.isstructured(A)
            p.A = factorize(A)
        elseif !(A isa AbstractDiffEqOperator)
            # Most likely QR is the one that is overloaded
            # Works on things like CuArrays
            p.A = qr(A)
        end
    end

    if A isa Union{
        Matrix,
        SymTridiagonal,
        Tridiagonal,
        Symmetric,
        Hermitian,
        ForwardSensitivityJacobian,
    } # No 2-arg form for SparseArrays!
        copyto!(x, b)
        ldiv!(p.A, x)
        # Missing a little bit of efficiency in a rare case
    elseif ArrayInterface.isstructured(A) || A isa SparseMatrixCSC
        ldiv!(x, p.A, b)
    elseif A isa AbstractDiffEqOperator ||
           (is_cuda_available() && A isa CUDA.CuArray)
        reltol = checkreltol(reltol)
        solver = LinSolveGMRES(abstol = 1f-16, reltol = reltol, kwargs...)
        solver(x, A, b)
    else
        ldiv!(x, p.A, b)
    end
    return nothing
end

function (p::DefaultLinSolve)(::Type{Val{:init}}, f, u0_prototype)
    if has_Wfact(f) || has_Wfact_t(f)
        piv = collect(
            one(LinearAlgebra.BlasInt):convert(
                LinearAlgebra.BlasInt,
                length(u0_prototype),
            ),
        ) # pivoting vector
        DefaultLinSolve(f, piv, nothing)
    else
        DefaultLinSolve()
    end
end

Base.resize!(p::DefaultLinSolve, i) = p.A = nothing
