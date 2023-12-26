# Specialize QR for the non-square case
# Missing ldiv! definitions: https://github.com/JuliaSparse/SparseArrays.jl/issues/242
function _ldiv!(x::Vector,
    A::Union{SparseArrays.QR, LinearAlgebra.QRCompactWY,
        SparseArrays.SPQR.QRSparse,
        SparseArrays.CHOLMOD.Factor}, b::Vector)
    x .= A \ b
end

function _ldiv!(x::AbstractVector,
    A::Union{SparseArrays.QR, LinearAlgebra.QRCompactWY,
        SparseArrays.SPQR.QRSparse,
        SparseArrays.CHOLMOD.Factor}, b::AbstractVector)
    x .= A \ b
end

# Ambiguity removal
_ldiv!(::SVector, 
       A::Union{SparseArrays.CHOLMOD.Factor, LinearAlgebra.QR, LinearAlgebra.QRCompactWY, SparseArrays.SPQR.QRSparse}, 
       b::AbstractVector) = (A \ b)
_ldiv!(::SVector, A::Union{SparseArrays.CHOLMOD.Factor, LinearAlgebra.QR, LinearAlgebra.QRCompactWY, SparseArrays.SPQR.QRSparse}, 
       b::SVector) = (A \ b)
