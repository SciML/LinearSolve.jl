## Preconditioners

scaling_preconditioner(s::Number) = I * (1/s), I * s
scaling_preconditioner(s::AbstractVector) = Diagonal(inv.(s)),Diagonal(s)

struct ComposePreconditioner{Ti,To}
    inner::Ti
    outer::To
end

Base.eltype(A::ComposePreconditioner) = promote_type(eltype(A.inner), eltype(A.outer))
Base.adjoint(A::ComposePreconditioner) = ComposePreconditioner(A.outer', A.inner')
Base.inv(A::ComposePreconditioner) = InvComposePreconditioner(A)

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

struct InvComposePreconditioner{Tp <: ComposePreconditioner}
    P::Tp
end

InvComposePreconditioner(inner, outer) = InvComposePreconditioner(ComposePreconditioner(inner, outer))

Base.eltype(A::InvComposePreconditioner) = Base.eltype(A.P)
Base.adjoint(A::InvComposePreconditioner) = InvComposePreconditioner(A.P')
Base.inv(A::InvComposePreconditioner) = deepcopy(A.P)

function LinearAlgebra.mul!(y, A::InvComposePreconditioner, x)
    @unpack P = A
    ldiv!(y, P, x)
end

function get_preconditioner(Pi, Po)

    ifPi = Pi !== Identity()
    ifPo = Po !== Identity()

    P =
    if ifPi & ifPo
        ComposePreconditioner(Pi, Po)
    elseif ifPi | ifPo
        ifPi ? Pi : Po
    else
        Identity()
    end

    return P
end
