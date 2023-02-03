# Tooling Preconditioners
#
# TODO - update Preconditioner docs
# TODO - replace ComposePreconditoner with ComposedOperator after
#        ComposePreconditioner is deprecated in OrdinaryDiffEq

#const ComposePreconditioner = ComposedOperator
#@deprecate ComposePreconditioner ComposedOperator

struct ComposePreconditioner{Ti, To}
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

#const InvPreconditioner = InvertedOperator
#@deprecate InvPreconditioner InvertedOperator
