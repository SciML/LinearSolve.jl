"""
    solve(prob::BorderedLinearProblem, alg)

Solve the bordered system in one shot. Use `init` plus `solve!` instead when the
same `J` is solved against repeatedly, which is the case this package is for.
"""
SciMLBase.solve(prob::BorderedLinearProblem, alg; kwargs...) =
    solve!(SciMLBase.init(prob, alg; kwargs...))

SciMLBase.solve(prob::BorderedLinearProblem; kwargs...) =
    SciMLBase.solve(prob, BorderingBLS(); kwargs...)
