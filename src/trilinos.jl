module TrilinosBindings
    using CxxWrap
    using SciMLBase

    @wrapmodule(() -> joinpath(@__DIR__, "../deps/libamesos2_wrap.dylib"))
    
    function __init__()
        @initcxx
    end

    struct Amesos2Solver <: SciMLBase.AbstractLinearAlgorithm end

    function SciMLBase.init(prob, alg::Amesos2Solver, args...; kwargs...)
        return prob
    end

    function SciMLBase.solve!(cache, alg::Amesos2Solver; kwargs...)
        dummy_solve()
        return cache
    end
end
