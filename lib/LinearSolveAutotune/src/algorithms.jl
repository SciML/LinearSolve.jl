# Algorithm detection and creation functions

"""
    get_available_algorithms()

Returns a list of available LU factorization algorithms based on the system and loaded packages.
"""
function get_available_algorithms()
    algs = []
    alg_names = String[]

    # Core algorithms always available
    push!(algs, LUFactorization())
    push!(alg_names, "LUFactorization")

    push!(algs, GenericLUFactorization())
    push!(alg_names, "GenericLUFactorization")

    # MKL if available
    if LinearSolve.usemkl
        push!(algs, MKLLUFactorization())
        push!(alg_names, "MKLLUFactorization")
    end

    # Apple Accelerate if available  
    if LinearSolve.appleaccelerate_isavailable()
        push!(algs, AppleAccelerateLUFactorization())
        push!(alg_names, "AppleAccelerateLUFactorization")
    end

    # RecursiveFactorization if loaded
    try
        if LinearSolve.userecursivefactorization(nothing)
            push!(algs, RFLUFactorization())
            push!(alg_names, "RFLUFactorization")
        end
    catch
        # RFLUFactorization not available
    end

    # SimpleLU always available
    push!(algs, SimpleLUFactorization())
    push!(alg_names, "SimpleLUFactorization")

    return algs, alg_names
end

"""
    get_gpu_algorithms()

Returns GPU-specific algorithms if GPU hardware and packages are available.
"""
function get_gpu_algorithms()
    gpu_algs = []
    gpu_names = String[]

    # CUDA algorithms
    if is_cuda_available()
        try
            push!(gpu_algs, CudaOffloadFactorization())
            push!(gpu_names, "CudaOffloadFactorization")
        catch
            # CUDA extension not loaded
        end
    end

    # Metal algorithms for Apple Silicon
    if is_metal_available()
        try
            push!(gpu_algs, MetalLUFactorization())
            push!(gpu_names, "MetalLUFactorization")
        catch
            # Metal extension not loaded
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
