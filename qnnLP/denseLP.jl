export denseLP

function denseLP(m::JuMP.Model, x::Union{VarOrAff, Array{VarOrAff}},
                     weights::Array{T, 2}, bias::Array{U, 1},
                     upper::Array{V, 1}, lower::Array{W, 1};
                     actFunc="")
    
end
