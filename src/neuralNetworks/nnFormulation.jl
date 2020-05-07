include("layers.jl")
export nnFormulation!

# Add constraints such that y = NN(x).
function nnFormulation!(model::JuMP.Model, nn, x::VarOrAff,
    y::VarOrAff; ideal=true)
    # TODO: Build the MIP formulation.
end
