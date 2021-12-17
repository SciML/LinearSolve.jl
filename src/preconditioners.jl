## Diagonal Preconditioners

struct DiagonalPreconditioner{D}
    diag::D
end
struct InvDiagonalPreconditioner{D}
    diag::D
end

Base.eltype(A::Union{DiagonalPreconditioner,InvDiagonalPreconditioner}) = eltype(A.diag)

function LinearAlgebra.ldiv!(A::DiagonalPreconditioner, x)
    x .= x ./ A.diag
end

function LinearAlgebra.ldiv!(y, A::DiagonalPreconditioner, x)
    y .= x ./ A.diag
end

#=
function LinearAlgebra.ldiv!(y::Matrix, A::DiagonalPreconditioner, b::Matrix)
    @inbounds @simd for j ∈ 1:size(y, 2)
        for i ∈ 1:length(A.diag)
            y[i,j] = b[i,j] / A.diag[i]
        end
    end
    return y
end
=#

function LinearAlgebra.ldiv!(A::InvDiagonalPreconditioner, x)
    x .= x .* A.diag
end

function LinearAlgebra.ldiv!(y, A::InvDiagonalPreconditioner, x)
    y .= x .* A.diag
end

#=
function LinearAlgebra.ldiv!(y::Matrix, A::InvDiagonalPreconditioner, b::Matrix)
    @inbounds @simd for j ∈ 1:size(y, 2)
        for i ∈ 1:length(A.diag)
            y[i,j] = b[i,j] * A.diag[i]
        end
    end
    return y
end
=#

LinearAlgebra.mul!(y, A::DiagonalPreconditioner, x) = LinearAlgebra.ldiv!(y, InvDiagonalPreconditioner(A.diag), x)
LinearAlgebra.mul!(y, A::InvDiagonalPreconditioner, x) = LinearAlgebra.ldiv!(y, DiagonalPreconditioner(A.diag), x)

## Compose Preconditioner

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
