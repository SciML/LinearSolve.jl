"""
    BorderedLinearProblem(J, b, c, d, f, g)

The bordered system

```
┌      ┐┌   ┐   ┌   ┐
│ J  b ││ v │ = │ f │
│ c' d ││ σ │   │ g │
└      ┘└   ┘   └   ┘
```

`f` is the state, and may be any array. For a 2D problem it is natural to keep the
state as a matrix, and nothing here requires it to be flattened into a vector.

`b` and `c` live in the same space as the state, so for a width-1 border they have the
same shape as `f`, and `d` and `g` are scalars.

For a border of width `m > 1`, pass `b` and `c` as a tuple (or vector) of `m` states,
`d` as an `m x m` matrix and `g` as a length-`m` vector. The width is taken from that
structure and never inferred from an array's shape, since a state can itself be a
matrix and the two cannot be told apart otherwise.

`J` is only ever applied and solved against, so it can be a matrix or a matrix-free
operator; no assumption is made about its shape.
"""
struct BorderedLinearProblem{TJ, Tb, Tc, Td, Tf, Tg}
    J::TJ
    b::Tb
    c::Tc
    d::Td
    f::Tf
    g::Tg

    function BorderedLinearProblem(J, b, c, d, f, g)
        mb, mc = _border_width(b), _border_width(c)
        mb == mc || throw(
            DimensionMismatch("b has border width $mb but c has $mc")
        )
        # Compare against the state rather than against `size(J, 1)`: `J` may be
        # matrix-free, and the state may have any shape.
        for (name, x) in (("b", b), ("c", c))
            for (i, xi) in enumerate(_border_parts(x))
                size(xi) == size(f) || throw(
                    DimensionMismatch(
                        "$name[$i] has size $(size(xi)), expected $(size(f)) to match the state"
                    )
                )
            end
        end
        _check_border_rhs(g, mb)
        return new{
            typeof(J), typeof(b), typeof(c), typeof(d), typeof(f), typeof(g),
        }(J, b, c, d, f, g)
    end
end

# A single state of any shape is a width-1 border. A tuple, or a vector whose elements
# are themselves arrays, is an explicit wide border.
_border_width(b) = 1
_border_width(b::Tuple) = length(b)
_border_width(b::AbstractVector{<:AbstractArray}) = length(b)

_border_parts(b) = (b,)
_border_parts(b::Tuple) = b
_border_parts(b::AbstractVector{<:AbstractArray}) = b

_check_border_rhs(g, m) =
    m == 1 || length(g) == m || throw(
    DimensionMismatch("g has length $(length(g)), expected $m to match the border width")
)

"""
    BorderedLinearSolution(v, σ, retcode)

`v` solves the state block and has the same shape as the state. `σ` solves the border
block, a scalar for a width-1 border. `retcode` is the `ReturnCode` of the underlying
`LinearSolve` solve, so a failure in the inner solver is visible rather than absorbed.
"""
struct BorderedLinearSolution{Tv, Tσ}
    v::Tv
    σ::Tσ
    retcode::SciMLBase.ReturnCode.T
end

SciMLBase.successful_retcode(sol::BorderedLinearSolution) =
    SciMLBase.successful_retcode(sol.retcode)
