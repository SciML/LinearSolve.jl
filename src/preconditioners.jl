# Tooling Preconditioners

"""
    ComposePreconditioner{Ti, To}

A preconditioner that composes two preconditioners by applying them sequentially.
The inner preconditioner is applied first, followed by the outer preconditioner.
This allows for building complex preconditioning strategies by combining simpler ones.

## Fields
- `inner::Ti`: The inner (first) preconditioner to apply
- `outer::To`: The outer (second) preconditioner to apply

## Usage

```julia
# Compose a diagonal preconditioner with an ILU preconditioner
inner_prec = DiagonalPreconditioner(diag(A))
outer_prec = ILUFactorization()  
composed = ComposePreconditioner(inner_prec, outer_prec)
```

The composed preconditioner applies: `outer(inner(x))` for any vector `x`.

## Mathematical Interpretation

For a linear system `Ax = b`, if `P₁` is the inner and `P₂` is the outer preconditioner,
then the composed preconditioner effectively applies `P₂P₁` as the combined preconditioner.
"""
struct ComposePreconditioner{Ti, To}
    inner::Ti
    outer::To
end

Base.eltype(A::ComposePreconditioner) = promote_type(eltype(A.inner), eltype(A.outer))

function LinearAlgebra.ldiv!(A::ComposePreconditioner, x)
    @unpack inner, outer = A

    ldiv!(inner, x)
    ldiv!(outer, x)
end

function LinearAlgebra.ldiv!(y, A::ComposePreconditioner, x)
    @unpack inner, outer = A

    ldiv!(y, inner, x)
    ldiv!(outer, y)
end

"""
    InvPreconditioner{T}

A preconditioner wrapper that treats a matrix or operator as if it represents
the inverse of the actual preconditioner. Instead of solving `Px = y`, it 
computes `P*y` where `P` is stored as the "inverse" preconditioner matrix.

## Fields
- `P::T`: The stored preconditioner matrix/operator (representing `P⁻¹`)

## Usage

This is useful when you have a matrix that approximates the inverse of your
desired preconditioner. For example, if you have computed an approximate 
inverse matrix `Ainv ≈ A⁻¹`, you can use:

```julia
prec = InvPreconditioner(Ainv)
```

## Mathematical Interpretation

For a linear system `Ax = b` with preconditioner `M`, normally we solve `M⁻¹Ax = M⁻¹b`.
With `InvPreconditioner`, the stored matrix `P` represents `M⁻¹` directly, so
applying the preconditioner becomes a matrix-vector multiplication rather than
a linear solve.

## Methods

- `ldiv!(A::InvPreconditioner, x)`: Computes `x ← P*x` (in-place)
- `ldiv!(y, A::InvPreconditioner, x)`: Computes `y ← P*x`  
- `mul!(y, A::InvPreconditioner, x)`: Computes `y ← P⁻¹*x` (inverse operation)
"""
struct InvPreconditioner{T}
    P::T
end

Base.eltype(A::InvPreconditioner) = Base.eltype(A.P)
LinearAlgebra.ldiv!(A::InvPreconditioner, x) = mul!(x, A.P, x)
LinearAlgebra.ldiv!(y, A::InvPreconditioner, x) = mul!(y, A.P, x)
LinearAlgebra.mul!(y, A::InvPreconditioner, x) = ldiv!(y, A.P, x)
