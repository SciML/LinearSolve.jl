# Algorithm detection and creation functions

"""
    get_available_algorithms(; skip_missing_algs::Bool = false)

Returns a list of available LU factorization algorithms based on the system and loaded packages.
If skip_missing_algs=false, errors when expected algorithms are missing; if true, warns instead.
"""
function get_available_algorithms(; skip_missing_algs::Bool = false)
    algs = []
    alg_names = String[]

    # Core algorithms always available
    push!(algs, LUFactorization())
    push!(alg_names, "LUFactorization")

    push!(algs, GenericLUFactorization())
    push!(alg_names, "GenericLUFactorization")

    if blis_jll.is_available()
        push!(algs, LinearSolve.BLISLUFactorization())
        push!(alg_names, "BLISLUFactorization")
    else
        @warn "blis.jll not available for this platform. BLISLUFactorization will not be included."
    end

    # MKL if available
    if LinearSolve.usemkl
        push!(algs, MKLLUFactorization())
        push!(alg_names, "MKLLUFactorization")
    end

    # Apple Accelerate if available (should be available on macOS)
    if LinearSolve.appleaccelerate_isavailable()
        push!(algs, AppleAccelerateLUFactorization())
        push!(alg_names, "AppleAccelerateLUFactorization")
    else
        # Check if we're on macOS and Apple Accelerate should be available
        if Sys.isapple() && !skip_missing_algs
            msg = "macOS system detected but Apple Accelerate not available. This is unexpected."
            @warn msg
        end
    end

    # RecursiveFactorization - should always be available as it's a hard dependency
    try
        if LinearSolve.userecursivefactorization(nothing)
            push!(algs, RFLUFactorization())
            push!(alg_names, "RFLUFactorization")
        else
            msg = "RFLUFactorization should be available (RecursiveFactorization.jl is a hard dependency)"
            if skip_missing_algs
                @warn msg
            else
                error(msg * ". Pass `skip_missing_algs=true` to continue with warning instead.")
            end
        end
    catch e
        msg = "RFLUFactorization failed to load: $e"
        if skip_missing_algs
            @warn msg
        else
            error(msg * ". Pass `skip_missing_algs=true` to continue with warning instead.")
        end
    end

    # SimpleLU always available
    push!(algs, SimpleLUFactorization())
    push!(alg_names, "SimpleLUFactorization")

    return algs, alg_names
end

"""
    get_gpu_algorithms(; skip_missing_algs::Bool = false)

Returns GPU-specific algorithms if GPU hardware and packages are available.
If skip_missing_algs=false, errors when GPU hardware is detected but algorithms are missing; if true, warns instead.
"""
function get_gpu_algorithms(; skip_missing_algs::Bool = false)
    gpu_algs = []
    gpu_names = String[]

    # CUDA algorithms
    if is_cuda_available()
        try
            push!(gpu_algs, CudaOffloadFactorization())
            push!(gpu_names, "CudaOffloadFactorization")
        catch e
            msg = "CUDA hardware detected but CudaOffloadFactorization not available: $e. Load CUDA.jl package."
            if skip_missing_algs
                @warn msg
            else
                error(msg * " Pass `skip_missing_algs=true` to continue with warning instead.")
            end
        end
    end

    # Metal algorithms for Apple Silicon
    if is_metal_available()
        try
            push!(gpu_algs, MetalLUFactorization())
            push!(gpu_names, "MetalLUFactorization")
        catch e
            msg = "Metal hardware detected but MetalLUFactorization not available: $e. Load Metal.jl package."
            if skip_missing_algs
                @warn msg
            else
                error(msg * " Pass `skip_missing_algs=true` to continue with warning instead.")
            end
        end
    end

    return gpu_algs, gpu_names
end

"""
    luflop(m, n=m; innerflop=2)

Calculate the number of floating point operations for LU factorization.
From the existing LinearSolve benchmarks.
"""
function luflop(m, n = m; innerflop = 2)
    sum(1:min(m, n)) do k
        invflop = 1
        scaleflop = isempty((k + 1):m) ? 0 : sum((k + 1):m)
        updateflop = isempty((k + 1):n) ? 0 :
                     sum((k + 1):n) do j
            isempty((k + 1):m) ? 0 : sum((k + 1):m) do i
                innerflop
            end
        end
        invflop + scaleflop + updateflop
    end
end
