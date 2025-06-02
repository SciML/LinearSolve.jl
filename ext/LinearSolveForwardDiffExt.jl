module LinearSolveForwardDiffExt 

const DualLinearProblem = LinearProblem{
    <:Union{Number, <:AbstractArray}, iip,
    <:Union{<:Dual{T,V,P}, <:AbstractArray{<:Dual{T,V,P}}}, 
    <:Union{<:Dual{T,V,P}, <:AbstractArray{<:Dual{T,V,P}}}, 
    <:Union{Number, <:AbstractArray}
} where {iip, T, V}


const DualALinearProblem = LinearProblem{
    <:Union{Number, <:AbstractArray}, 
    iip,
    <:Union{<:Dual{T, V, P}, <:AbstractArray{<:Dual{T, V, P}}}, 
    <:Union{Number, <:AbstractArray},
    <:Union{Number, <:AbstractArray}
}

const DualBLinearProblem = LinearProblem{
    <:Union{Number, <:AbstractArray}, 
    iip,
    <:Union{Number, <:AbstractArray}, 
    <:Union{<:Dual{T, V, P}, <:AbstractArray{<:Dual{T, V, P}}},
    <:Union{Number, <:AbstractArray}
}

const DualAbstractLinearProblem = Union{DualLinearProblem, DualALinearProblem, DualBLinearProblem}


function linearsolve_forwarddiff_solve(prob::LinearProblem, alg, args...; kwargs...)
    new_A = nodual_value(prob.A)
    new_b = nodual_value(prob.b)

    newprob = remake(prob; A = new_A, b = new_b)

    sol = solve(newprob, alg, args...; kwargs...)
    uu = sol.u

    ∂_A = partial_vals(A)
    ∂_b = partial_vals(b)

    rhs = xp_linsolve_rhs(uu, ∂_A, ∂_b)

    partial_prob = remake(newprob, b = rhs)
    partial_sol = solve(partial_prob, alg, args...; kwargs...)

    sol, partial_sol
end



partial_vals(x::Dual) = ForwardDiff.partials(x)
partial_vals(x::AbstractArray{<:Dual}) = map(ForwardDiff.value, x)
partial_vals(x) = nothing

nodual_value(x) = x
nodual_value(x::Dual) = ForwardDiff.value(x)
nodual_value(x::AbstractArray{<:Dual}) = map(ForwardDiff.value, x)


function xp_linsolve_rhs(uu, ∂_A::Union{<:Partials, <:AbstractArray{<:Partials}}, ∂_b::Union{<:Partials, <:AbstractArray{<:Partials}})
    A_list = partials_to_list(∂_A)
    b_list = partials_to_list(∂_b) 

    Auu = [A*uu for A in A_list]

    reduce(hcat, b_list .- Auu)
end

function xp_linsolve_rhs(uu, ∂_A::Union{<:Partials, <:AbstractArray{<:Partials}}, ∂_b::Nothing)
    A_list = partials_to_list(∂_A)

    Auu = [A*uu for A in A_list]

    reduce(hcat, Auu)
end

function xp_linsolve_rhs(uu, ∂_A::Nothing, ∂_b::Union{<:Partials, <:AbstractArray{<:Partials}})
    b_list = partials_to_list(∂_b)

    reduce(hcat, b_list)
end



function partials_to_list(partial_matrix::Vector)
    p = eachindex(first(partial_matrix))
    [[partial[i] for partial in partial_matrix]  for i in p]
end

function partials_to_list(partial_matrix)
    p = length(first(partial_matrix))
    m,n = size(partial_matrix)
    res_list = fill(zeros(m,n),p)
    for k in 1:p
        res = zeros(m,n)
        for i in 1:m
            for j in 1:n
                res[i,j] = partial_matrix[i,j][k]
            end
        end
        res_list[k] = res
    end
    return res_list
end










