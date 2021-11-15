struct KrylovJL{A,K} <: SciMLLinearSolveAlgorithm
    solver::Function
    args::A
    kwargs::K
end

function KrylovJL(args...; solver = gmres, kwargs...)
    return KrylovJL(solver, args, kwargs)
end

function SciMLBase.solve(cache::LinearCache, alg::KrylovJL,args...;kwargs...)
    @unpack A, b, u, Pr, Pl = cache
    u, stats = alg.solver(A, b, args...; M=Pl, N=Pr, kwargs...)
    resid = A * u - b
    retcode = stats.solved ? :Success : :Failure
    return u #SciMLBase.build_solution(prob, alg, x, resid; retcode = retcode)
end
