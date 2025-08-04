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

    # BLIS if JLL packages are available and hardware supports it
    try
        # Check if BLIS_jll and LAPACK_jll are available, which enable BLISLUFactorization
        blis_jll_available = haskey(Base.loaded_modules, Base.PkgId(Base.UUID("068f7417-6964-5086-9a5b-bc0c5b4f7fa6"), "BLIS_jll"))
        lapack_jll_available = haskey(Base.loaded_modules, Base.PkgId(Base.UUID("51474c39-65e3-53ba-86ba-03b1b862ec14"), "LAPACK_jll"))
        
        if (blis_jll_available || lapack_jll_available) && isdefined(LinearSolve, :BLISLUFactorization) && hasmethod(LinearSolve.BLISLUFactorization, ())
            # Test if BLIS works on this hardware
            try
                test_alg = LinearSolve.BLISLUFactorization()
                # Simple test to see if it can be created
                push!(algs, test_alg)
                push!(alg_names, "BLISLUFactorization")
            catch e
                msg = "BLISLUFactorization available but not supported on this hardware: $e"
                if skip_missing_algs
                    @warn msg
                else
                    @info msg  # BLIS hardware incompatibility is not an error, just info
                end
            end
        else
            if blis_jll_available || lapack_jll_available
                msg = "BLIS_jll/LAPACK_jll loaded but BLISLUFactorization not available in LinearSolve"
            else
                msg = "BLIS_jll and LAPACK_jll not loaded - BLISLUFactorization requires these JLL packages"
            end
            if skip_missing_algs
                @warn msg
            else
                @info msg  # Not having BLIS JLL packages is not an error
            end
        end
    catch e
        msg = "Error checking BLIS JLL package availability: $e"
        if skip_missing_algs
            @warn msg
        else
            @info msg
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
