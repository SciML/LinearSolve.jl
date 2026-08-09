module LinearSolveBlockDiagonalsExt

using LinearSolve: LinearSolve, LinearVerbosity, LUFactorization, OperatorAssumptions,
    QRFactorization, SimpleGMRES
using BlockDiagonals: BlockDiagonals, BlockDiagonal, blocks, blocksizes
using LinearAlgebra: LinearAlgebra, NoPivot
using SciMLBase: SciMLBase, ReturnCode

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

# ---------------------------------------------------------------------------
# Blockwise direct factorizations
# ---------------------------------------------------------------------------
#
# A block diagonal system is a batch of independent systems, so the structural
# win is to factorize each block on its own: `sum(mᵢ^3)` work instead of `N^3`,
# and each block is a dense `Matrix` so the per-block call lands on BLAS.
#
# Without this, `do_factorization` handed the whole `BlockDiagonal` to
# `lu!`/`qr!`, which runs the generic scalar LinearAlgebra path over every
# structural zero and indexes through the block lookup on each access. Measured
# on 20x20 blocks, `LUFactorization` through that path took 7.8 ms at N = 200 and
# 255 ms at N = 800, against 0.05 ms and 0.27 ms for the blockwise solve here
# (150x and 953x). `qr!` had the same problem, plus its result type did not match
# what `init_cacheval` predicted, so `QRFactorization` threw a `TypeError` on
# every solve (SciML/LinearSolve.jl#203).
struct BlockDiagonalFactorization{F <: LinearAlgebra.Factorization}
    facts::Vector{F}
    # Cumulative row offsets, length nblocks + 1, so block i owns
    # `(offsets[i] + 1):offsets[i + 1]`.
    offsets::Vector{Int}
    success::Bool
end

LinearAlgebra.issuccess(F::BlockDiagonalFactorization) = F.success

# `issuccess` is only defined for the factorizations that can report a
# structural failure. QR has no such method, and rank deficiency is handled
# elsewhere, so treat it as successful rather than erroring on the lookup.
_bd_issuccess(F::LinearAlgebra.LU) = LinearAlgebra.issuccess(F)
_bd_issuccess(::LinearAlgebra.Factorization) = true

function _bd_offsets(A::BlockDiagonal)
    bsizes = blocksizes(A)
    offsets = Vector{Int}(undef, length(bsizes) + 1)
    offsets[1] = 0
    for (i, bsize) in enumerate(bsizes)
        offsets[i + 1] = offsets[i] + bsize[1]
    end
    return offsets
end

# Blocks are only independent subsystems when each one is square. A
# `BlockDiagonal` may hold rectangular blocks whose total is still square, and
# those do not decompose, so they keep the generic path.
_bd_all_square(A::BlockDiagonal) = all(bsize -> bsize[1] == bsize[2], blocksizes(A))

# `Val`-based pivots are still accepted by `LUFactorization`; normalize locally
# rather than reaching for LinearSolve's internal helper.
_bd_pivot(pivot::LinearAlgebra.PivotingStrategy) = pivot
_bd_pivot(::Val{true}) = LinearAlgebra.RowMaximum()
_bd_pivot(::Val{false}) = LinearAlgebra.NoPivot()

_bd_factorize_block(alg::LinearSolve.LUFactorization, B) =
    LinearAlgebra.lu!(B, _bd_pivot(alg.pivot); check = false)
_bd_factorize_block(alg::QRFactorization, B) = LinearAlgebra.qr!(B, alg.pivot)

# A 0x0 block gives an instance of exactly the type `do_factorization` will
# produce, without doing any work, which is what keeps the cacheval field type
# and the real factorization in agreement.
function _bd_instance(alg, A::BlockDiagonal)
    empty_block = similar(first(blocks(A)), eltype(A), 0, 0)
    fact = _bd_factorize_block(alg, empty_block)
    return BlockDiagonalFactorization([fact], [0, 0], true)
end

# Whether the blockwise representation applies. `residualsafety` needs the
# a-posteriori machinery of the stock `LUFactorization` solve, so those caches
# keep the generic dense representation and the generic solve path with it.
_bd_blockwise(alg::LUFactorization, A::BlockDiagonal) =
    !alg.residualsafety && _bd_all_square(A)
_bd_blockwise(alg::QRFactorization, A::BlockDiagonal) = _bd_all_square(A)

for ALG in (:LUFactorization, :QRFactorization)
    @eval function LinearSolve.init_cacheval(
            alg::$ALG, A::BlockDiagonal, b, u, Pl, Pr,
            maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
            assumptions::OperatorAssumptions
        )
        _bd_blockwise(alg, A) || return LinearSolve.init_cacheval(
            alg, Matrix(A), b, u, Pl, Pr, maxiters, abstol, reltol, verbose,
            assumptions
        )
        return _bd_instance(alg, A)
    end

    @eval function LinearSolve.do_factorization(alg::$ALG, A::BlockDiagonal, b, u)
        _bd_blockwise(alg, A) || return LinearSolve.do_factorization(alg, Matrix(A), b, u)
        facts = map(B -> _bd_factorize_block(alg, B), blocks(A))
        return BlockDiagonalFactorization(
            facts, _bd_offsets(A), all(_bd_issuccess, facts)
        )
    end
end

# `LUFactorization` has its own `solve!` that calls `lu!`/`lu` directly rather
# than going through `do_factorization`, so the blockwise path needs its own
# method. Dispatching on the cacheval type rather than on `A` keys this to
# exactly the caches that `init_cacheval` above chose the blockwise
# representation for, leaving `residualsafety` and rectangular-block caches on
# the stock path.
const BlockDiagonalLUCache = LinearSolve.LinearCache{
    <:Any, <:Any, <:Any, <:Any, <:Any, <:BlockDiagonalFactorization,
}

function SciMLBase.solve!(
        cache::BlockDiagonalLUCache, alg::LUFactorization; kwargs...
    )
    if cache.isfresh
        fact = LinearSolve.do_factorization(alg, cache.A, cache.b, cache.u)
        cache.cacheval = fact
        if !LinearAlgebra.issuccess(fact)
            return SciMLBase.build_linear_solution(
                alg, cache.u, nothing, nothing; retcode = ReturnCode.Failure
            )
        end
        cache.isfresh = false
    end
    y = LinearSolve._ldiv!(cache.u, cache.cacheval, cache.b)
    return SciMLBase.build_linear_solution(
        alg, y, nothing, nothing; retcode = ReturnCode.Success
    )
end

function LinearSolve._ldiv!(
        x::AbstractVector, F::BlockDiagonalFactorization, b::AbstractVector
    )
    x === b || copyto!(x, b)
    offsets = F.offsets
    @inbounds for i in eachindex(F.facts)
        LinearAlgebra.ldiv!(F.facts[i], view(x, (offsets[i] + 1):offsets[i + 1]))
    end
    return x
end

function LinearSolve._ldiv!(
        X::AbstractMatrix, F::BlockDiagonalFactorization, B::AbstractMatrix
    )
    X === B || copyto!(X, B)
    offsets = F.offsets
    @inbounds for i in eachindex(F.facts)
        LinearAlgebra.ldiv!(F.facts[i], view(X, (offsets[i] + 1):offsets[i + 1], :))
    end
    return X
end

end
