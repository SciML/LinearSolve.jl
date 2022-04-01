struct FunctionCall{F,A} <: SciMLLinearSolveAlgorithm
    func!::F
    argsyms::A

    function FunctionCall(func!::Function, argsyms::Tuple)
        new{typeof(func!), typeof(argsyms)}(func!, argsyms)
    end
end

function (f::FunctionCall)(cache::LinearCache)
    @unpack func!, argsyms = f
    args = [getproperty(cache,argsym) for argsym in argsyms]
    func!(args...)
end

function SciMLBase.solve(cache::LinearCache, alg::FunctionCall,
                         args...; kwargs...)
    @unpack u, b = cache
    copy!(u, b)
    alg(cache)

    return SciMLBase.build_linear_solution(alg,cache.u,nothing,cache)
end
