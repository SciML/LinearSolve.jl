struct KrylovJL{AT,bT,A,K} <: SciMLLinearSolveAlgorithm
    solver::Function
    A::AT
    b::bT
    args::A
    kwargs::K
end

function KrylovJL(A, b, args...; solver = gmres, kwargs...)
    return KrylovJL(solver, A, b, args, kwargs)
end

function SciMLBase.solve(cache::LinearCache, alg::KrylovJL,args...;kwargs...)
    @unpack A, b = cache
    x, stats = alg.solver(A, b, args...; kwargs...)
    resid = A * x - b
    retcode = stats.solved ? :Success : :Failure
    return x #SciMLBase.build_solution(prob, alg, x, resid; retcode = retcode)
end
