function __init__()
    @static if VERSION < v"1.7beta"
        blas = BLAS.vendor()
        IS_OPENBLAS[] = blas == :openblas64 || blas == :openblas
    else
        IS_OPENBLAS[] = occursin("openblas", BLAS.get_config().loaded_libs[1].libname)
    end
end
