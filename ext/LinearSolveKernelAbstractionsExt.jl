module LinearSolveKernelAbstractionsExt

using LinearSolve, KernelAbstractions

LinearSolve.__is_extension_loaded(::Val{:KernelAbstractions}) = true

using GPUArraysCore

function LinearSolve._fast_sym_givens!(c, s, R, nr::Int, inner_iter::Int, bsize::Int, Hbis)
    backend = get_backend(Hbis)
    kernel! = __fast_sym_givens_kernel!(backend)
    kernel!(c[inner_iter], s[inner_iter], R[nr + inner_iter], Hbis; ndrange=bsize)
    return c, s, R
end

@kernel function __fast_sym_givens_kernel!(c, s, R, @Const(Hbis))
    idx = @index(Global)
    @inbounds _c, _s, _ρ = LinearSolve._sym_givens(R[idx], Hbis[idx])
    @inbounds c[idx] = _c
    @inbounds s[idx] = _s
    @inbounds R[idx] = _ρ
end

end
