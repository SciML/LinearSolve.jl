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

"""
    get_detailed_system_info()

Returns a comprehensive DataFrame with detailed system information suitable for CSV export.
Includes versioninfo() details and hardware-specific information for analysis.
"""
function get_detailed_system_info()
    # Basic system information
    system_data = Dict{String, Any}()
    
    # Julia and system basics
    system_data["timestamp"] = string(Dates.now())
    system_data["julia_version"] = string(VERSION)
    system_data["julia_commit"] = Base.GIT_VERSION_INFO.commit[1:10]  # Short commit hash
    system_data["os_name"] = Sys.iswindows() ? "Windows" : Sys.islinux() ? "Linux" : Sys.isapple() ? "macOS" : "Other"
    system_data["os_version"] = string(Sys.KERNEL)
    system_data["architecture"] = string(Sys.ARCH)
    system_data["cpu_cores"] = Sys.CPU_THREADS
    system_data["julia_threads"] = Threads.nthreads()
    system_data["word_size"] = Sys.WORD_SIZE
    system_data["machine"] = Sys.MACHINE
    
    # CPU details
    cpu_info = Sys.cpu_info()[1]
    system_data["cpu_name"] = cpu_info.model
    system_data["cpu_speed_mhz"] = cpu_info.speed
    
    # Categorize CPU vendor for easy analysis
    cpu_name_lower = lowercase(system_data["cpu_name"])
    if contains(cpu_name_lower, "intel")
        system_data["cpu_vendor"] = "Intel"
    elseif contains(cpu_name_lower, "amd")
        system_data["cpu_vendor"] = "AMD"
    elseif contains(cpu_name_lower, "apple") || contains(cpu_name_lower, "m1") || contains(cpu_name_lower, "m2") || contains(cpu_name_lower, "m3")
        system_data["cpu_vendor"] = "Apple"
    else
        system_data["cpu_vendor"] = "Other"
    end
    
    # BLAS and linear algebra libraries
    system_data["blas_vendor"] = string(LinearAlgebra.BLAS.vendor())
    # LAPACK vendor detection (safe for different Julia versions)
    try
        system_data["lapack_vendor"] = string(LinearAlgebra.LAPACK.vendor())
    catch
        # Fallback: LAPACK vendor often matches BLAS vendor
        system_data["lapack_vendor"] = system_data["blas_vendor"]
    end
    system_data["blas_num_threads"] = LinearAlgebra.BLAS.get_num_threads()
    
    # LinearSolve-specific package availability
    system_data["mkl_available"] = LinearSolve.usemkl
    system_data["mkl_used"] = system_data["mkl_available"] && contains(lowercase(system_data["blas_vendor"]), "mkl")
    system_data["apple_accelerate_available"] = LinearSolve.appleaccelerate_isavailable()
    system_data["apple_accelerate_used"] = system_data["apple_accelerate_available"] && contains(lowercase(system_data["blas_vendor"]), "accelerate")
    
    # BLIS availability check
    system_data["blis_available"] = false
    system_data["blis_used"] = false
    try
        # Check if BLIS is loaded and BLISLUFactorization is available
        if isdefined(LinearSolve, :BLISLUFactorization) && hasmethod(LinearSolve.BLISLUFactorization, ())
            system_data["blis_available"] = true
            # Check if BLIS is actually being used (contains "blis" in BLAS vendor)
            system_data["blis_used"] = contains(lowercase(system_data["blas_vendor"]), "blis")
        end
    catch
        # If there's any error checking BLIS, leave as false
    end
    
    # GPU information
    system_data["cuda_available"] = is_cuda_available()
    system_data["metal_available"] = is_metal_available()
    
    # Try to detect if CUDA/Metal packages are actually loaded
    system_data["cuda_loaded"] = false
    system_data["metal_loaded"] = false
    try
        # Check if CUDA algorithms are actually available
        if system_data["cuda_available"]
            system_data["cuda_loaded"] = isdefined(Main, :CUDA) || haskey(Base.loaded_modules, Base.PkgId(Base.UUID("052768ef-5323-5732-b1bb-66c8b64840ba"), "CUDA"))
        end
        if system_data["metal_available"]
            system_data["metal_loaded"] = isdefined(Main, :Metal) || haskey(Base.loaded_modules, Base.PkgId(Base.UUID("dde4c033-4e86-420c-a63e-0dd931031962"), "Metal"))
        end
    catch
        # If we can't detect, leave as false
    end
    
    # Environment information
    system_data["libm"] = Base.libm_name
    # libdl_name may not exist in all Julia versions
    try
        system_data["libdl"] = Base.libdl_name
    catch
        system_data["libdl"] = "unknown"
    end
    
    # Memory information (if available)
    try
        if Sys.islinux()
            meminfo = read(`cat /proc/meminfo`, String)
            mem_match = match(r"MemTotal:\s*(\d+)\s*kB", meminfo)
            if mem_match !== nothing
                system_data["total_memory_gb"] = round(parse(Int, mem_match.captures[1]) / 1024 / 1024, digits=2)
            else
                system_data["total_memory_gb"] = missing
            end
        elseif Sys.isapple()
            mem_bytes = parse(Int, read(`sysctl -n hw.memsize`, String))
            system_data["total_memory_gb"] = round(mem_bytes / 1024^3, digits=2)
        else
            system_data["total_memory_gb"] = missing
        end
    catch
        system_data["total_memory_gb"] = missing
    end
    
    # Create DataFrame with single row
    return DataFrame([system_data])
end
