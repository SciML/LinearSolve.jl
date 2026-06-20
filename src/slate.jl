"""
    SLATEFactorization(; libpath = nothing)

Dense LU solve using SLATE's LAPACK compatibility API.

This calls the `slate_*gesv` entry points from `libslate_lapack_api` and supports
`Float32`, `Float64`, `ComplexF32`, and `ComplexF64` systems. Other real and
complex element types are promoted to `Float64` and `ComplexF64`, respectively.

The SLATE library is loaded lazily. If `libpath` is not provided, LinearSolve
checks `ENV["SLATE_LAPACK_LIB"]`, `ENV["SLATE_LIB"]`, then standard library
names such as `libslate_lapack_api.so` and `libslate_lapack_api.dylib`.
"""
struct SLATEFactorization{L} <: AbstractDenseFactorization
    libpath::L
end

SLATEFactorization(; libpath = nothing) = SLATEFactorization(libpath)

mutable struct SLATECache{T}
    A::Matrix{T}
    B::Matrix{T}
    ipiv::Vector{Cint}
end

const _SLATE_LIB = Ref{Union{Nothing, Ptr{Cvoid}}}(nothing)
const _SLATE_LIBPATH = Ref{Union{Nothing, String}}(nothing)
const _SLATE_SYMBOLS = Dict{Tuple{Ptr{Cvoid}, Symbol}, Ptr{Cvoid}}()

function _slate_eltype(A, b)
    T = promote_type(eltype(A), eltype(b))
    if T <: Float32
        return Float32
    elseif T <: Float64
        return Float64
    elseif T <: ComplexF32
        return ComplexF32
    elseif T <: Complex
        return ComplexF64
    elseif T <: Real
        return Float64
    else
        error("SLATEFactorization supports real and complex numeric matrices")
    end
end

function _slate_library_candidates(libpath)
    candidates = String[]
    libpath === nothing || push!(candidates, String(libpath))
    for key in ("SLATE_LAPACK_LIB", "SLATE_LIB")
        if haskey(ENV, key) && !isempty(ENV[key])
            push!(candidates, ENV[key])
        end
    end
    append!(
        candidates,
        (
            "libslate_lapack_api.so",
            "libslate_lapack_api.dylib",
            "slate_lapack_api",
        ),
    )
    return unique(candidates)
end

function _load_libslate(libpath = nothing)
    requested = libpath === nothing ? nothing : String(libpath)
    if _SLATE_LIB[] !== nothing && (requested === nothing || requested == _SLATE_LIBPATH[])
        return _SLATE_LIB[]
    end

    for candidate in _slate_library_candidates(libpath)
        handle = Libdl.dlopen_e(candidate)
        if handle != C_NULL
            _SLATE_LIB[] = handle
            _SLATE_LIBPATH[] = candidate
            empty!(_SLATE_SYMBOLS)
            return handle
        end
    end

    return nothing
end

function slate_isavailable(; libpath = nothing)
    lib = _load_libslate(libpath)
    lib === nothing && return false
    try
        _slate_symbol(lib, :slate_dgesv)
        return true
    catch
        return false
    end
end

function _slate_lib(alg::SLATEFactorization)
    lib = _load_libslate(alg.libpath)
    lib !== nothing && return lib
    error(
        "SLATEFactorization requires SLATE's LAPACK API library. " *
        "Build SLATE with `slate_lapack_api`, then pass `SLATEFactorization(libpath = \"/path/to/libslate_lapack_api.so\")` " *
        "or set ENV[\"SLATE_LAPACK_LIB\"]."
    )
end

function _slate_symbol(lib::Ptr{Cvoid}, basename::Symbol)
    key = (lib, basename)
    cached = get(_SLATE_SYMBOLS, key, C_NULL)
    cached != C_NULL && return cached

    names = (
        basename,
        Symbol(basename, :_),
        Symbol(uppercase(String(basename))),
        Symbol(uppercase(String(basename)), :_),
    )
    for name in names
        ptr = Libdl.dlsym_e(lib, name)
        if ptr != C_NULL
            _SLATE_SYMBOLS[key] = ptr
            return ptr
        end
    end

    error("SLATE library does not export $(basename) or a supported Fortran-mangled variant")
end

_slate_gesv_symbol(::Type{Float32}) = :slate_sgesv
_slate_gesv_symbol(::Type{Float64}) = :slate_dgesv
_slate_gesv_symbol(::Type{ComplexF32}) = :slate_cgesv
_slate_gesv_symbol(::Type{ComplexF64}) = :slate_zgesv

function init_cacheval(
        ::SLATEFactorization, A, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    )
    T = _slate_eltype(A, b)
    nrhs = b isa AbstractVector ? 1 : size(b, 2)
    return SLATECache(Matrix{T}(undef, size(A, 1), size(A, 2)),
        Matrix{T}(undef, size(A, 1), nrhs), Vector{Cint}(undef, min(size(A)...)))
end

function _resize_slate_cache!(scache::SLATECache{T}, A, b) where {T}
    n = size(A, 1)
    nrhs = b isa AbstractVector ? 1 : size(b, 2)
    size(scache.A) == size(A) || (scache.A = Matrix{T}(undef, size(A)...))
    size(scache.B) == (n, nrhs) || (scache.B = Matrix{T}(undef, n, nrhs))
    length(scache.ipiv) == min(size(A)...) || resize!(scache.ipiv, min(size(A)...))
    return scache
end

function _copy_slate_rhs!(B::Matrix{T}, b::AbstractVector) where {T}
    @views copyto!(B[:, 1], b)
    return B
end

function _copy_slate_rhs!(B::Matrix{T}, b::AbstractMatrix) where {T}
    copyto!(B, b)
    return B
end

function _slate_solution!(u::AbstractVector, B::AbstractMatrix)
    if size(B, 2) == 1 && length(u) == length(B)
        copyto!(u, vec(B))
        return u
    else
        return copy(B)
    end
end

function _slate_solution!(u::AbstractMatrix, B::AbstractMatrix)
    copyto!(u, B)
    return u
end

function _slate_retcode(info::Cint)
    if info == 0
        return ReturnCode.Success
    elseif info > 0
        return ReturnCode.Failure
    else
        throw(ArgumentError("SLATE GESV received an invalid argument at position $(-info)"))
    end
end

function SciMLBase.solve!(cache::LinearCache, alg::SLATEFactorization; kwargs...)
    A = convert(AbstractMatrix, cache.A)
    size(A, 1) == size(A, 2) || error("SLATEFactorization requires a square matrix")
    size(A, 1) == size(cache.b, 1) || error("SLATEFactorization RHS length does not match A")

    T = _slate_eltype(A, cache.b)
    scache = @get_cacheval(cache, :SLATEFactorization)
    if !(scache isa SLATECache{T})
        scache = init_cacheval(
            alg, A, cache.b, cache.u, cache.Pl, cache.Pr, cache.maxiters,
            cache.abstol, cache.reltol, cache.verbose, cache.assumptions
        )
        cache.cacheval = scache
    end
    _resize_slate_cache!(scache, A, cache.b)

    copyto!(scache.A, A)
    _copy_slate_rhs!(scache.B, cache.b)

    n = Ref{Cint}(size(scache.A, 1))
    nrhs = Ref{Cint}(size(scache.B, 2))
    lda = Ref{Cint}(max(1, stride(scache.A, 2)))
    ldb = Ref{Cint}(max(1, stride(scache.B, 2)))
    info = Ref{Cint}(0)
    lib = _slate_lib(alg)
    fptr = _slate_symbol(lib, _slate_gesv_symbol(T))

    ccall(
        fptr,
        Cvoid,
        (Ref{Cint}, Ref{Cint}, Ptr{Cvoid}, Ref{Cint}, Ptr{Cint}, Ptr{Cvoid}, Ref{Cint}, Ref{Cint}),
        n,
        nrhs,
        pointer(scache.A),
        lda,
        scache.ipiv,
        pointer(scache.B),
        ldb,
        info,
    )

    retcode = _slate_retcode(info[])
    y = cache.u
    if retcode == ReturnCode.Success
        y = _slate_solution!(cache.u, scache.B)
    end
    cache.isfresh = false

    return SciMLBase.build_linear_solution(alg, y, nothing, cache; retcode)
end
