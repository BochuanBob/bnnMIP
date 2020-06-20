using JuMP
const VarOrAff = Union{JuMP.VariableRef,JuMP.AffExpr,
                        Array{VariableRef}, Array{AffExpr}}
using Combinatorics, IterTools
export initNN!, aff_callback_value

mutable struct nnData
    count::Int
    nnData() = new(0)
end

mutable struct timeData
    time::Float64
    timeData() = new(0)
end

# In order to add base_name for variables.
function initNN!(m::JuMP.Model)
    if !haskey(m.ext, :NN)
        m.ext[:NN] = nnData()
    end
    if !haskey(m.ext, :CUTS)
        m.ext[:CUTS] = nnData()
    end
    if !haskey(m.ext, :BENCH_CONV2D)
        m.ext[:BENCH_CONV2D] = timeData()
    end
    if !haskey(m.ext, :CALLBACK_TIME)
        m.ext[:CALLBACK_TIME] = timeData()
    end
    if !haskey(m.ext, :TEST_CONSTRAINTS)
        m.ext[:TEST_CONSTRAINTS] = nnData()
    end
    return nothing
end

function aff_callback_value(cb_data, aff::GenericAffExpr{Float64,VariableRef})
   ret = aff.constant
   for (var, coef) in aff.terms
       ret += coef * callback_value(cb_data, var)
   end
   return ret
end

function aff_callback_value(cb_data, aff::VariableRef)
   return JuMP.callback_value(cb_data, aff)
end
