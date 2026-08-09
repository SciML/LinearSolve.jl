module LinearSolveBlockDiagonalsExt

using LinearSolve: LinearSolve, LinearVerbosity, OperatorAssumptions,
    QRFactorization, SimpleGMRES
using BlockDiagonals: BlockDiagonals, BlockDiagonal, blocks, blocksizes
using LinearAlgebra: LinearAlgebra, NoPivot

function LinearSolve.init_cacheval(
        alg::SimpleGMRES{false}, A::BlockDiagonal, b, args...;
        kwargs...
    )
    @assert ndims(A) == 2 "ndims(A) == $(ndims(A)). `A` must have ndims == 2."
    # We need to perform this check even when `zeroinit == true`, since the type of the
    # cache is dependent on whether we are able to use the specialized dispatch.
    bsizes = blocksizes(A)
    usize = first(first(bsizes))
    uniform_blocks = true
    for bsize in bsizes
        if bsize[1] != usize || bsize[2] != usize
            uniform_blocks = false
            break
        end
    end
    # Can't help but perform dynamic dispatch here
    return LinearSolve._init_cacheval(
        Val(uniform_blocks), alg, A, b, args...;
        blocksize = usize, kwargs...
    )
end

# `qr!` of a `BlockDiagonal` stays block diagonal, so it runs the generic
# LinearAlgebra path in place and returns a `QR` wrapping the `BlockDiagonal`
# itself. The default `init_cacheval` instead predicts the type through
# `ArrayInterface.qr_instance`, which densifies and reports `QRCompactWY`;
# assigning the real factorization into that differently-typed cacheval field
# then threw a `TypeError` on every solve. Build the instance from the block
# type so the predicted and actual types agree.
#
# Only `NoPivot` is supported: column pivoting has to move entries across
# blocks, which `BlockDiagonal` rejects ("Cannot set entry ... in
# off-diagonal-block to nonzero value"), so `QRFactorization(ColumnNorm())`
# cannot represent its result in this storage type at all.
function LinearSolve.init_cacheval(
        alg::QRFactorization{NoPivot}, A::BlockDiagonal, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    )
    # A 0x0 block keeps the instance allocation-free while preserving the
    # `BlockDiagonal` wrapper; `similar(A, T, 0, 0)` would densify to a `Matrix`
    # and reintroduce the mismatch this method exists to remove.
    empty_block = similar(first(blocks(A)), eltype(A), 0, 0)
    return LinearAlgebra.QR(
        BlockDiagonal([empty_block]),
        similar(A, eltype(A), 0)
    )
end

end
