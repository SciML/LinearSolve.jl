struct ForwardSensitivityJacobian{T,JJ<:AbstractMatrix{T}} <: AbstractMatrix{T}
    J::JJ
end

Base.parent(J::ForwardSensitivityJacobian) = J.J

Base.similar(J::ForwardSensitivityJacobian, ::Type{T}) where {T} =
    ForwardSensitivityJacobian(similar(parent(J), T))

struct ForwardSensitivityJacobianFactorization{T,F<:Factorization{T}} <:
       Factorization{T}
    factorization::F
end

LinearAlgebra.lu!(J::ForwardSensitivityJacobian) =
    ForwardSensitivityJacobianFactorization(lu!(parent(J)))

function LinearAlgebra.ldiv!(F::ForwardSensitivityJacobianFactorization, x)
    F = F.factorization
    n = size(F, 1)
    k = length(x) ÷ n
    @assert k * n == length(x)
    ldiv!(F, reshape(x, n, k))
    return x
end
