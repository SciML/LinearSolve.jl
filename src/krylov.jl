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

function SciMLBase.solve(prob::LinearProblem, alg::KrylovJL, args...; kwargs...)
    @unpack A, b, p = prob
    x, stats = alg.solver(A, b, args...; kwargs...)
    resid = mul!(similar(b), A, x) - b
    retcode = stats.solved ? :Success : :Failure
    return SciMLBase.build_solution(prob, alg, x, resid; retcode = retcode)
end