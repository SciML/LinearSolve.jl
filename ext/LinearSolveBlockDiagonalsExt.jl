module LinearSolveBlockDiagonalsExt

using LinearSolve, BlockDiagonals

function LinearSolve.init_cacheval(alg::SimpleGMRES{false}, A::BlockDiagonal, b, args...;
        kwargs...)
    @assert ndims(A)==2 "ndims(A) == $(ndims(A)). `A` must have ndims == 2."
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
    return LinearSolve._init_cacheval(Val(uniform_blocks), alg, A, b, args...;
        blocksize = usize, kwargs...)
end

end
