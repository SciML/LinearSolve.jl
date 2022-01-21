isgpu(x) = false
ifgpufree(x) = nothing
function __init__()
    @static if VERSION < v"1.7beta"
      blas = BLAS.vendor()
      IS_OPENBLAS[] = blas == :openblas64 || blas == :openblas
    else
      IS_OPENBLAS[] = occursin("openblas", BLAS.get_config().loaded_libs[1].libname)
    end

    @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" include("cuda.jl")
    @require Pardiso="46dd5b70-b6fb-5a00-ae2d-e8fea33afaf2" include("pardiso.jl")
end
