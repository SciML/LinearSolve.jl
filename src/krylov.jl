struct KrylovJL{A,K} <: SciMLLinearSolveAlgorithm
    solver::Function
    args::A
    kwargs::K
end

function KrylovJL(args...; solver = gmres, kwargs...)
    return KrylovJL(solver, args, kwargs)
end

function SciMLBase.solve(cache::LinearCache, alg::KrylovJL,args...;kwargs...)
    @unpack A, b, Pl,Pr = cache
    x, stats = alg.solver(A, b, args...; M=Pl, N=Pr, kwargs...)
    resid = A * x - b
    retcode = stats.solved ? :Success : :Failure
    return x #SciMLBase.build_solution(prob, alg, x, resid; retcode = retcode)
end
