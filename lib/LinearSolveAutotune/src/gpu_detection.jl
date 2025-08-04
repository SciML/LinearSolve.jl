# GPU hardware and package detection

"""
    is_cuda_available()

Check if CUDA hardware and packages are available.
"""
function is_cuda_available()
    # Check if CUDA extension is loaded
    ext = Base.get_extension(LinearSolve, :LinearSolveCUDAExt)
    if ext === nothing
        return false
    end

    # Check if we have CUDA.jl loaded
    try
        CUDA = Base.get_extension(LinearSolve, :LinearSolveCUDAExt).CUDA
        return CUDA.functional()
    catch
        return false
    end
end

"""
    is_metal_available()

Check if Metal (Apple Silicon) hardware and packages are available.
"""
function is_metal_available()
    # Check if we're on macOS with Apple Silicon
    if !Sys.isapple()
        return false
    end

    # Check if Metal extension is loaded
    ext = Base.get_extension(LinearSolve, :LinearSolveMetalExt)
    if ext === nothing
        return false
    end

    # Check if we have Metal.jl loaded and functional
    try
        Metal = Base.get_extension(LinearSolve, :LinearSolveMetalExt).Metal
        return Metal.functional()
    catch
        return false
    end
end

"""
    get_system_info()

Get system information for telemetry reporting.
"""
function get_system_info()
    info = Dict{String, Any}()

    info["julia_version"] = string(VERSION)
    info["os"] = string(Sys.KERNEL)
    info["arch"] = string(Sys.ARCH)
    info["cpu_name"] = Sys.cpu_info()[1].model
    info["num_cores"] = Sys.CPU_THREADS
    info["num_threads"] = Threads.nthreads()
    info["blas_vendor"] = string(LinearAlgebra.BLAS.vendor())
    info["has_cuda"] = is_cuda_available()
    info["has_metal"] = is_metal_available()

    if LinearSolve.usemkl
        info["mkl_available"] = true
    else
        info["mkl_available"] = false
    end

    if LinearSolve.appleaccelerate_isavailable()
        info["apple_accelerate_available"] = true
    else
        info["apple_accelerate_available"] = false
    end

    return info
end
