module LinearSolveRecursiveArrayToolsExt

using LinearSolve, RecursiveArrayTools
import LinearSolve: init_cacheval

# Krylov.jl tries to init with `ArrayPartition(undef, ...)`. Avoid hitting that!
function init_cacheval(alg::LinearSolve.KrylovJL, A, b::ArrayPartition, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Bool, ::LinearSolve.OperatorAssumptions)
    return nothing
end

end
