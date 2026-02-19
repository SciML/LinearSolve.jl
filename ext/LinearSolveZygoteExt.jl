module LinearSolveZygoteExt

using Zygote
using SciMLBase

# Zygote ignores ChainRulesCore.rrule for Base.getindex on AbstractArray subtypes
# (https://github.com/FluxML/Zygote.jl/issues/811). Since LinearSolution <: AbstractArray
# (via AbstractNoTimeSolution), Zygote's hardcoded AbstractArray getindex adjoint would
# shadow the CRC rule. We register an explicit Zygote adjoint to override it.
Zygote.@adjoint function Base.getindex(sol::SciMLBase.LinearSolution, i::Integer)
    function LinearSolution_getindex_pullback(Δ)
        du = zero(sol.u)
        du[i] = Δ
        Δsol = SciMLBase.build_linear_solution(sol.cache.alg, du, sol.resid, sol.cache)
        return (Δsol, nothing)
    end
    return sol[i], LinearSolution_getindex_pullback
end

end
