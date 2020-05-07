using JuMP
const VarOrAff = Union{JuMP.VariableRef,JuMP.AffExpr}

mutable struct nnData
    count::Int
    nnData() = new(0)
end

# In order to add base_name for variables.
function initNN!(m::JuMP.Model)
    if !haskey(m.ext, :ROT)
        m.ext[:NN] = nnData()
    end
    return nothing
end
