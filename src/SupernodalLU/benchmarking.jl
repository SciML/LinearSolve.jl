"""
    supernodal_panel_solve!(W, Y, np; algorithm = :kernel, sweep = :lower)

Apply one SupernodalLU pivot-panel triangular solve in place. `algorithm` selects
`:kernel`, `:triangularsolve`, or `:blas`; `sweep` selects `:lower` or `:upper`.
This is intended for benchmarking the panel-solve implementation choices.
"""
function supernodal_panel_solve!(
        W::AbstractMatrix, Y::AbstractMatrix, np::Integer;
        algorithm::Symbol = :kernel, sweep::Symbol = :lower
    )
    return _supernodal_panel_solve!(W, Y, Int(np), Val(algorithm), Val(sweep))
end

function _supernodal_panel_solve!(W, Y, np, ::Val{:kernel}, ::Val{:lower})
    return _unit_lower_solve!(W, Y, np)
end

function _supernodal_panel_solve!(W, Y, np, ::Val{:kernel}, ::Val{:upper})
    return _upper_solve!(W, Y, np)
end

function _supernodal_panel_solve!(W, Y, np, ::Val{:blas}, ::Val{:lower})
    return _panel_unit_lower_trsm!(W, Y, np)
end

function _supernodal_panel_solve!(W, Y, np, ::Val{:blas}, ::Val{:upper})
    return _panel_upper_trsm!(W, Y, np)
end

function _supernodal_panel_solve!(W, Y, np, ::Val{:triangularsolve}, sweep)
    throw(ArgumentError("TriangularSolve must be loaded to benchmark the triangularsolve path"))
end

function _supernodal_panel_solve!(W, Y, np, algorithm, sweep)
    throw(ArgumentError("unsupported SupernodalLU panel algorithm or sweep: $(algorithm), $(sweep)"))
end
