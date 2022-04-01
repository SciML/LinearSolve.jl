struct FunctionCall{F,A,K} <: SciMLLinearSolveAlgorithm
    func::F
    args::A
    kwargs::K

    function FunctionCall(func::Function, args::Tuple; kwargs...)
        @assert iscallable(func)
        @assert applicable(func, args; kwargs)

        new{typeof(func), typeof(args), typeof(kwargs)}(func, args, kwargs)
    end
end

function init_cacheval(alg::FunctionCall, cache::LinearCache)
    cache.cacheval
end

function SciMLBase.solve(cache::LinearCache, alg::FunctionCall,
                         args...; kwargs...)

    return
end
