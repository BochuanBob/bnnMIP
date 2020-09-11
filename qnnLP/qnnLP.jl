using JuMP, Gurobi
using MAT
using Random
using Combinatorics, IterTools
const VarOrAff = Union{JuMP.VariableRef,JuMP.AffExpr}
export initNN!

mutable struct nnData
    count::Int
    nnData() = new(0)
end

# In order to add base_name for variables.
function initNN!(m::JuMP.Model)
    if !haskey(m.ext, :NN)
        m.ext[:NN] = nnData()
    end
    return nothing
end


include("denseLP.jl")
include("flatten.jl")
include("utilities.jl")
include("verification.jl")
