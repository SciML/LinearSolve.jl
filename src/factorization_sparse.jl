# Specialize QR for the non-square case
# Missing ldiv! definitions: https://github.com/JuliaSparse/SparseArrays.jl/issues/242
function _ldiv!(x::Vector,
                A::Union{SparseArrays.QR, LinearAlgebra.QRCompactWY,
                         SuiteSparse.SPQR.QRSparse}, b::Vector)
    x .= A \ b
end
