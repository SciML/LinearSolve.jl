
""" user passes in inverse function, and arg symbols """
struct FunctionCall{F,A} <: SciMLLinearSolveAlgorithm
    func!::F
    argnames::A

    function FunctionCall(func!::Function, argnames::Tuple)
#       @assert hasfield(::LinearCache, argnames)
#       @assert isdefined
        new{typeof(func!), typeof(argnames)}(func!, argnames)
    end
end

function (f::FunctionCall)(cache::LinearCache)
    @unpack func!, argnames = f
    args = [getproperty(cache,argname) for argname in argnames]
    func!(args...)
end

function SciMLBase.solve(cache::LinearCache, alg::FunctionCall,
                         args...; kwargs...)
    @unpack u, b = cache
    alg(cache)
    return SciMLBase.build_linear_solution(alg,cache.u,nothing,cache)
end

##

""" apply ldiv!(A, u) """
struct ApplyLDivBang2Args <: SciMLLinearSolveAlgorithm end
function SciMLBase.solve(cache::LinearCache, alg::ApplyLDivBang2Args,
                         args...; kwargs...)
    @unpack A, b, u = cache
    copy!(u, b)
    LinearAlgebra.ldiv!(A, u)
    return SciMLBase.build_linear_solution(alg,cache.u,nothing,cache)
end

""" apply ldiv!(u, A, b) """
struct ApplyLDivBang3Args <: SciMLLinearSolveAlgorithm end
function SciMLBase.solve(cache::LinearCache, alg::ApplyLDivBang3Args,
                         args...; kwargs...)
    @unpack A, b, u = cache
    LinearAlgebra.ldiv!(u, A, b)
    return SciMLBase.build_linear_solution(alg,cache.u,nothing,cache)
end
