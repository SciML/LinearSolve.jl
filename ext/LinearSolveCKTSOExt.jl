module LinearSolveCKTSOExt

using CKTSO: CKTSO, cktso, cktso!
using LinearSolve: LinearSolve, CKTSOFactorization, LinearVerbosity, OperatorAssumptions
using SciMLBase: SciMLBase, ReturnCode
using SparseArrays: SparseArrays, AbstractSparseMatrixCSC, SparseMatrixCSC, nonzeros

# CKTSO keeps the symbolic analysis and refactorizes the same pattern with new values,
# which is the whole reason to reach for it, so the cache holds the factorization and the
# pattern it was built from rather than rebuilding per solve.
mutable struct CKTSOCache{F}
    fact::F
end

_csc(A::SparseMatrixCSC{Float64, <:Integer}) = A
_csc(A::AbstractSparseMatrixCSC) = SparseMatrixCSC{Float64, Int64}(A)
_csc(A) = SparseMatrixCSC{Float64, Int64}(A)

function LinearSolve.init_cacheval(
        ::CKTSOFactorization, A, b, u, Pl, Pr, maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    # Nothing is factorized here: CKTSO's analysis is the expensive part and `solve!`
    # does it once, on the matrix it is actually given.
    return CKTSOCache{Union{Nothing, CKTSO.CKTSOSolver{Float64}}}(nothing)
end

function SciMLBase.solve!(
        cache::LinearSolve.LinearCache, alg::CKTSOFactorization; kwargs...
    )
    A = convert(AbstractMatrix, cache.A)
    size(A, 1) == size(A, 2) || error("CKTSOFactorization requires a square matrix")

    ccache = LinearSolve.@get_cacheval(cache, :CKTSOFactorization)
    A_csc = _csc(A)

    try
        if cache.isfresh
            if ccache.fact === nothing
                ccache.fact = cktso(A_csc; threads = alg.threads)
            else
                # Same pattern is the common case in the workload CKTSO targets, so try
                # the cheap refactorization and fall back to a fresh analysis when the
                # pattern really did change.
                try
                    cktso!(ccache.fact, A_csc)
                catch err
                    err isa ArgumentError || rethrow()
                    ccache.fact = cktso(A_csc; threads = alg.threads)
                end
            end
            cache.isfresh = false
        end

        copyto!(cache.u, cache.b)
        CKTSO.ldiv!(cache.u, ccache.fact, cache.b)

        return SciMLBase.build_linear_solution(
            alg, cache.u, nothing, cache; retcode = ReturnCode.Success
        )
    catch err
        err isa CKTSO.CKTSOError || rethrow()
        # A singular matrix is a fact about the problem, not an internal failure, so it
        # comes back as a retcode the caller can branch on. `-3` is in this list because
        # CKTSO reports an empty column that way, and the CSC arrays handed to it are
        # well formed by construction, so a square matrix is the only way to get it.
        if err.code == -3 || err.code == -5 || err.code == -6
            return SciMLBase.build_linear_solution(
                alg, cache.u, nothing, cache; retcode = ReturnCode.Failure
            )
        end
        rethrow()
    end
end

end
