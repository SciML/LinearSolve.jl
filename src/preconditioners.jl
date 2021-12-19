# Tooling Preconditioners

struct ComposePreconditioner{Ti,To}
    inner::Ti
    outer::To
end

Base.eltype(A::ComposePreconditioner) = promote_type(eltype(A.inner), eltype(A.outer))

function LinearAlgebra.ldiv!(A::ComposePreconditioner, x)
    @unpack inner, outer = A

    ldiv!(inner, x)
    ldiv!(outer, x)
end

function LinearAlgebra.ldiv!(y, A::ComposePreconditioner, x)
    @unpack inner, outer = A

    ldiv!(y, inner, x)
    ldiv!(outer, y)
end

struct InvPreconditioner{T}
    P::T
end

Base.eltype(A::InvPreconditioner) = Base.eltype(A.P)
LinearAlgebra.mul!(y, A::InvPreconditioner, x) = ldiv!(y, A.P, x)
