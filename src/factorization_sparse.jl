# Specialize QR for the non-square case
# Missing ldiv! definitions: https://github.com/JuliaSparse/SparseArrays.jl/issues/242
function _ldiv!(x::Vector,
                A::Union{SparseArrays.QR, LinearAlgebra.QRCompactWY,
                         SuiteSparse.SPQR.QRSparse}, b::Vector)
    x .= A \ b
end

struct SparspakFactorization <: AbstractFactorization end

function init_cacheval(::SparspakFactorization, A, b, u, Pl, Pr, maxiters::Int, abstol,
                       reltol,
                       verbose::Bool, assumptions::OperatorAssumptions)
    A = convert(AbstractMatrix, A)
    p = Sparspak.Problem.Problem(size(A)...)
    Sparspak.Problem.insparse!(p, A)
    Sparspak.Problem.infullrhs!(p, b)
    s = Sparspak.SparseSolver.SparseSolver(p)
    return s
end

function SciMLBase.solve(cache::LinearCache, alg::SparspakFactorization; kwargs...)
    A = cache.A
    A = convert(AbstractMatrix, A)
    if cache.isfresh
        p = Sparspak.Problem.Problem(size(A)...)
        Sparspak.Problem.insparse!(p, A)
        Sparspak.Problem.infullrhs!(p, cache.b)
        s = Sparspak.SparseSolver.SparseSolver(p)
        cache = set_cacheval(cache, s)
    end
    Sparspak.SparseSolver.solve!(cache.cacheval)
    copyto!(cache.u, cache.cacheval.p.x)
    SciMLBase.build_linear_solution(alg, cache.u, nothing, cache)
end