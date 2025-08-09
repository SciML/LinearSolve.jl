# GPU hardware and package detection

using CPUSummary
using Pkg

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
    info["os_name"] = Sys.iswindows() ? "Windows" :
                      Sys.islinux() ? "Linux" : Sys.isapple() ? "macOS" : "Other"
    info["arch"] = string(Sys.ARCH)

    # Use CPUSummary where available, fallback to Sys otherwise
    try
        info["cpu_name"] = string(Sys.CPU_NAME)
    catch
        info["cpu_name"] = "Unknown"
    end

    # CPUSummary.num_cores() returns the physical cores (as Static.StaticInt)
    info["num_cores"] = Int(CPUSummary.num_cores())
    info["num_logical_cores"] = Sys.CPU_THREADS
    info["num_threads"] = Threads.nthreads()

    # BLAS threads
    try
        info["blas_num_threads"] = LinearAlgebra.BLAS.get_num_threads()
    catch
        info["blas_num_threads"] = 1
    end

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

    # Add package versions
    info["package_versions"] = get_package_versions()

    return info
end

"""
    get_package_versions()

Get versions of LinearSolve-related packages and their dependencies.
Returns a Dict with package names and versions.
"""
function get_package_versions()
    versions = Dict{String, String}()

    # Get the current project's dependencies
    deps = Pkg.dependencies()

    # List of packages we're interested in tracking
    important_packages = [
        "LinearSolve",
        "LinearSolveAutotune",
        "RecursiveFactorization",
        "CUDA",
        "Metal",
        "MKL_jll",
        "BLISBLAS",
        "AppleAccelerate",
        "SparseArrays",
        "KLU",
        "Pardiso",
        "MKLPardiso",
        "BandedMatrices",
        "FastLapackInterface",
        "HYPRE",
        "IterativeSolvers",
        "Krylov",
        "KrylovKit",
        "LinearAlgebra"
    ]

    # Also track JLL packages for BLAS libraries
    jll_packages = [
        "MKL_jll",
        "OpenBLAS_jll",
        "OpenBLAS32_jll",
        "blis_jll",
        "LAPACK_jll",
        "CompilerSupportLibraries_jll"
    ]

    all_packages = union(important_packages, jll_packages)

    # Iterate through dependencies and collect versions
    for (uuid, dep) in deps
        if dep.name in all_packages
            if dep.version !== nothing
                versions[dep.name] = string(dep.version)
            else
                # Try to get version from the package itself if loaded
                try
                    pkg_module = Base.loaded_modules[Base.PkgId(uuid, dep.name)]
                    if isdefined(pkg_module, :version)
                        versions[dep.name] = string(pkg_module.version)
                    else
                        versions[dep.name] = "unknown"
                    end
                catch
                    versions[dep.name] = "unknown"
                end
            end
        end
    end

    # Try to get Julia's LinearAlgebra stdlib version
    try
        versions["LinearAlgebra"] = string(VERSION)  # Stdlib version matches Julia
    catch
        versions["LinearAlgebra"] = "stdlib"
    end

    # Get BLAS configuration info
    try
        blas_config = LinearAlgebra.BLAS.get_config()
        if hasfield(typeof(blas_config), :loaded_libs)
            for lib in blas_config.loaded_libs
                if hasfield(typeof(lib), :libname)
                    lib_name = basename(string(lib.libname))
                    # Extract version info if available
                    versions["BLAS_lib"] = lib_name
                end
            end
        end
    catch
        # Fallback for older Julia versions
        versions["BLAS_vendor"] = string(LinearAlgebra.BLAS.vendor())
    end

    return versions
end

"""
    get_detailed_system_info()

Returns a comprehensive DataFrame with detailed system information suitable for CSV export.
Includes versioninfo() details and hardware-specific information for analysis.
"""
function get_detailed_system_info()
    # Basic system information
    system_data = Dict{String, Any}()

    # Julia and system basics - all with safe fallbacks
    try
        system_data["timestamp"] = string(Dates.now())
    catch
        system_data["timestamp"] = "unknown"
    end

    try
        system_data["julia_version"] = string(VERSION)
    catch
        system_data["julia_version"] = "unknown"
    end

    try
        system_data["julia_commit"] = Base.GIT_VERSION_INFO.commit[1:10]  # Short commit hash
    catch
        system_data["julia_commit"] = "unknown"
    end

    try
        system_data["os_name"] = Sys.iswindows() ? "Windows" :
                                 Sys.islinux() ? "Linux" : Sys.isapple() ? "macOS" : "Other"
    catch
        system_data["os_name"] = "unknown"
    end

    try
        system_data["os_version"] = string(Sys.KERNEL)
    catch
        system_data["os_version"] = "unknown"
    end

    try
        system_data["architecture"] = string(Sys.ARCH)
    catch
        system_data["architecture"] = "unknown"
    end

    try
        system_data["cpu_cores"] = Int(CPUSummary.num_cores())
    catch
        system_data["cpu_cores"] = "unknown"
    end

    try
        system_data["cpu_logical_cores"] = Sys.CPU_THREADS
    catch
        system_data["cpu_logical_cores"] = "unknown"
    end

    try
        system_data["julia_threads"] = Threads.nthreads()
    catch
        system_data["julia_threads"] = "unknown"
    end

    try
        system_data["word_size"] = Sys.WORD_SIZE
    catch
        system_data["word_size"] = "unknown"
    end

    try
        system_data["machine"] = Sys.MACHINE
    catch
        system_data["machine"] = "unknown"
    end

    # CPU details
    try
        system_data["cpu_name"] = string(Sys.CPU_NAME)
    catch
        system_data["cpu_name"] = "unknown"
    end

    try
        # Architecture info from Sys
        system_data["cpu_architecture"] = string(Sys.ARCH)
    catch
        system_data["cpu_architecture"] = "unknown"
    end

    # Categorize CPU vendor for easy analysis
    try
        cpu_name_lower = lowercase(string(system_data["cpu_name"]))
        if contains(cpu_name_lower, "intel")
            system_data["cpu_vendor"] = "Intel"
        elseif contains(cpu_name_lower, "amd")
            system_data["cpu_vendor"] = "AMD"
        elseif contains(cpu_name_lower, "apple") || contains(cpu_name_lower, "m1") ||
               contains(cpu_name_lower, "m2") || contains(cpu_name_lower, "m3")
            system_data["cpu_vendor"] = "Apple"
        else
            system_data["cpu_vendor"] = "Other"
        end
    catch
        system_data["cpu_vendor"] = "unknown"
    end

    # BLAS and linear algebra libraries
    try
        system_data["blas_vendor"] = string(LinearAlgebra.BLAS.vendor())
    catch
        system_data["blas_vendor"] = "unknown"
    end

    # LAPACK vendor detection (safe for different Julia versions)
    try
        system_data["lapack_vendor"] = string(LinearAlgebra.LAPACK.vendor())
    catch
        # Fallback: LAPACK vendor often matches BLAS vendor
        system_data["lapack_vendor"] = get(system_data, "blas_vendor", "unknown")
    end

    try
        system_data["blas_num_threads"] = LinearAlgebra.BLAS.get_num_threads()
    catch
        system_data["blas_num_threads"] = "unknown"
    end

    # LinearSolve-specific package availability
    try
        system_data["mkl_available"] = LinearSolve.usemkl
    catch
        system_data["mkl_available"] = false
    end

    try
        system_data["mkl_used"] = system_data["mkl_available"] &&
                                  contains(lowercase(string(system_data["blas_vendor"])), "mkl")
    catch
        system_data["mkl_used"] = false
    end

    try
        system_data["apple_accelerate_available"] = LinearSolve.appleaccelerate_isavailable()
    catch
        system_data["apple_accelerate_available"] = false
    end

    try
        system_data["apple_accelerate_used"] = system_data["apple_accelerate_available"] &&
                                               contains(
            lowercase(string(system_data["blas_vendor"])), "accelerate")
    catch
        system_data["apple_accelerate_used"] = false
    end

    # BLIS availability check - based on JLL packages
    system_data["blis_available"] = false
    system_data["blis_used"] = false
    system_data["blis_jll_loaded"] = false
    system_data["lapack_jll_loaded"] = false

    try
        # Check if BLIS_jll and LAPACK_jll are loaded
        system_data["blis_jll_loaded"] = haskey(Base.loaded_modules,
            Base.PkgId(Base.UUID("068f7417-6964-5086-9a5b-bc0c5b4f7fa6"), "BLIS_jll"))
        system_data["lapack_jll_loaded"] = haskey(Base.loaded_modules,
            Base.PkgId(Base.UUID("51474c39-65e3-53ba-86ba-03b1b862ec14"), "LAPACK_jll"))

        # BLIS is available if JLL packages are loaded and BLISLUFactorization exists
        if (system_data["blis_jll_loaded"] || system_data["lapack_jll_loaded"]) &&
           isdefined(LinearSolve, :BLISLUFactorization) &&
           hasmethod(LinearSolve.BLISLUFactorization, ())
            system_data["blis_available"] = true
            # Check if BLIS is actually being used (contains "blis" in BLAS vendor)
            system_data["blis_used"] = contains(lowercase(string(system_data["blas_vendor"])), "blis")
        end
    catch
        # If there's any error checking BLIS JLL packages, leave as false
    end

    # GPU information
    try
        system_data["cuda_available"] = is_cuda_available()
    catch
        system_data["cuda_available"] = false
    end

    try
        system_data["metal_available"] = is_metal_available()
    catch
        system_data["metal_available"] = false
    end

    # Try to detect if CUDA/Metal packages are actually loaded
    system_data["cuda_loaded"] = false
    system_data["metal_loaded"] = false
    try
        # Check if CUDA algorithms are actually available
        if system_data["cuda_available"]
            system_data["cuda_loaded"] = isdefined(Main, :CUDA) ||
                                         haskey(Base.loaded_modules,
                Base.PkgId(Base.UUID("052768ef-5323-5732-b1bb-66c8b64840ba"), "CUDA"))
        end
        if system_data["metal_available"]
            system_data["metal_loaded"] = isdefined(Main, :Metal) ||
                                          haskey(Base.loaded_modules,
                Base.PkgId(Base.UUID("dde4c033-4e86-420c-a63e-0dd931031962"), "Metal"))
        end
    catch
        # If we can't detect, leave as false
    end

    # Environment information
    try
        system_data["libm"] = Base.libm_name
    catch
        system_data["libm"] = "unknown"
    end

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
                system_data["total_memory_gb"] = round(
                    parse(Int, mem_match.captures[1]) / 1024 / 1024, digits = 2)
            else
                system_data["total_memory_gb"] = "unknown"
            end
        elseif Sys.isapple()
            mem_bytes = parse(Int, read(`sysctl -n hw.memsize`, String))
            system_data["total_memory_gb"] = round(mem_bytes / 1024^3, digits = 2)
        else
            system_data["total_memory_gb"] = "unknown"
        end
    catch
        system_data["total_memory_gb"] = "unknown"
    end

    # Create DataFrame with single row
    return DataFrame([system_data])
end
