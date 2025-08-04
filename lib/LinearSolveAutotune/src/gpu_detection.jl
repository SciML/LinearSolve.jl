# GPU hardware and package detection

"""
    is_cuda_available()

Check if CUDA hardware and packages are available.
Issues warnings if CUDA hardware is detected but packages aren't loaded.
"""
function is_cuda_available()
    # Check if CUDA extension is loaded
    ext = Base.get_extension(LinearSolve, :LinearSolveCUDAExt)
    if ext === nothing
        # Check if we might have CUDA hardware but missing packages
        try
            # Try to detect NVIDIA GPUs via nvidia-smi or similar system indicators
            if haskey(ENV, "CUDA_VISIBLE_DEVICES") ||
               (Sys.islinux() && isfile("/proc/driver/nvidia/version")) ||
               (Sys.iswindows() && success(`where nvidia-smi`))
                @warn "CUDA hardware may be available but CUDA.jl extension is not loaded. Consider adding `using CUDA` to enable GPU algorithms."
            end
        catch
            # Silently continue if detection fails
        end
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
Issues warnings if Metal hardware is detected but packages aren't loaded.
"""
function is_metal_available()
    # Check if we're on macOS with Apple Silicon
    if !Sys.isapple()
        return false
    end

    # Check if this is Apple Silicon
    is_apple_silicon = Sys.ARCH == :aarch64

    # Check if Metal extension is loaded
    ext = Base.get_extension(LinearSolve, :LinearSolveMetalExt)
    if ext === nothing
        if is_apple_silicon
            @warn "Apple Silicon hardware detected but Metal.jl extension is not loaded. Consider adding `using Metal` to enable GPU algorithms."
        end
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
