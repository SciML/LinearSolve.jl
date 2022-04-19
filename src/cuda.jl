gpu_or_cpu(x::CUDA.CuArray) = CUDA.CuArray
gpu_or_cpu(x::Transpose{<:Any,<:CUDA.CuArray}) = CUDA.CuArray
gpu_or_cpu(x::Adjoint{<:Any,<:CUDA.CuArray}) = CUDA.CuArray
isgpu(::CUDA.CuArray) = true
isgpu(::Transpose{<:Any,<:CUDA.CuArray}) = true
isgpu(::Adjoint{<:Any,<:CUDA.CuArray}) = true
ifgpufree(x::CUDA.CuArray) = CUDA.unsafe_free!(x)
ifgpufree(x::Transpose{<:Any,<:CUDA.CuArray}) = CUDA.unsafe_free!(x.parent)
ifgpufree(x::Adjoint{<:Any,<:CUDA.CuArray}) = CUDA.unsafe_free!(x.parent)

@require Tracker="9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c" begin
    TrackedArray = Tracker.TrackedArray
    gpu_or_cpu(x::TrackedArray{<:Any,<:Any,<:CUDA.CuArray}) = CUDA.CuArray
    gpu_or_cpu(x::Adjoint{<:Any,TrackedArray{<:Any,<:Any,<:CUDA.CuArray}}) = CUDA.CuArray
    gpu_or_cpu(x::Transpose{<:Any,TrackedArray{<:Any,<:Any,<:CUDA.CuArray}}) = CUDA.CuArray
    isgpu(::Adjoint{<:Any,TrackedArray{<:Any,<:Any,<:CUDA.CuArray}}) = true
    isgpu(::TrackedArray{<:Any,<:Any,<:CUDA.CuArray}) = true
    isgpu(::Transpose{<:Any,TrackedArray{<:Any,<:Any,<:CUDA.CuArray}}) = true
    ifgpufree(x::TrackedArray{<:Any,<:Any,<:CUDA.CuArray}) = CUDA.unsafe_free!(x.data)
    ifgpufree(x::Adjoint{<:Any,TrackedArray{<:Any,<:Any,<:CUDA.CuArray}}) = CUDA.unsafe_free!((x.data).parent)
    ifgpufree(x::Transpose{<:Any,TrackedArray{<:Any,<:Any,<:CUDA.CuArray}}) = CUDA.unsafe_free!((x.data).parent)
end

struct GPUOffloadFactorization <: AbstractFactorization end

function SciMLBase.solve(cache::LinearCache, alg::GPUOffloadFactorization; kwargs...)
    if cache.isfresh
        fact = do_factorization(alg, CUDA.CuArray(cache.A), cache.b, cache.u)
        cache = set_cacheval(cache, fact)
    end

    copyto!(cache.u,cache.b)
    y = Array(ldiv!(cache.cacheval, CUDA.CuArray(cache.u)))
    SciMLBase.build_linear_solution(alg,y,nothing,cache)
end

function do_factorization(alg::GPUOffloadFactorization, A, b, u)
    A isa Union{AbstractMatrix,AbstractDiffEqOperator} ||
        error("LU is not defined for $(typeof(A))")

    if A isa AbstractDiffEqOperator
        A = A.A
    end
    fact = qr(CUDA.CuArray(A))
    return fact
end

if VERSION <= v"1.7.2"

function LinearAlgebra.ldiv!(x::CUDA.CuArray,_qr::CUDA.CUSOLVER.CuQR,b::CUDA.CuArray)
  _x = UpperTriangular(_qr.R) \ (_qr.Q' * reshape(b,length(b),1))
  x .= vec(_x)
  CUDA.unsafe_free!(_x)
  return x
end
# make `\` work
LinearAlgebra.ldiv!(F::CUDA.CUSOLVER.CuQR, b::CUDA.CuArray) = (x = similar(b); ldiv!(x, F, b); x)

end

export GPUOffloadFactorization
