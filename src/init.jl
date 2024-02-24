function __init__()
    IS_OPENBLAS[] = occursin("openblas", BLAS.get_config().loaded_libs[1].libname)

    HAS_APPLE_ACCELERATE[] = __appleaccelerate_isavailable()
end
