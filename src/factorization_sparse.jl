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
function _ldiv!(::SVector,
        A::Union{SparseArrays.CHOLMOD.Factor, LinearAlgebra.QR,
            LinearAlgebra.QRCompactWY, SparseArrays.SPQR.QRSparse},
        b::AbstractVector)
    (A \ b)
end
function _ldiv!(::SVector,
        A::Union{SparseArrays.CHOLMOD.Factor, LinearAlgebra.QR,
            LinearAlgebra.QRCompactWY, SparseArrays.SPQR.QRSparse},
        b::SVector)
    (A \ b)
end

function pattern_changed(fact, A::SparseArrays.SparseMatrixCSC)
    !(SparseArrays.decrement(SparseArrays.getcolptr(A)) ==
    fact.colptr && SparseArrays.decrement(SparseArrays.getrowval(A)) ==
                 fact.rowval)
end