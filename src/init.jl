function __init__()
    @static if VERSION < v"1.7beta"
        blas = BLAS.vendor()
        IS_OPENBLAS[] = blas == :openblas64 || blas == :openblas
    else
        IS_OPENBLAS[] = occursin("openblas", BLAS.get_config().loaded_libs[1].libname)
    end
    @static if !isdefined(Base, :get_extension)
        @require IterativeSolvers="b77e0a4c-d291-57a0-90e8-8db25a27a240" begin
            include("../ext/LinearSolveIterativeSolversExt.jl")
        end
        @require KrylovKit="0b1a1467-8014-51b9-945f-bf0ae24f4b77" begin
            include("../ext/LinearSolveKrylovKitExt.jl")
        end
        @require Enzyme="7da242da-08ed-463a-9acd-ee780be4f1d9" begin
            include("../ext/LinearSolveEnzymeExt.jl")
        end
    end
end
