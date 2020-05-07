include("layerSetup.jl")
export flatten

# Take the input variables and ouput a flatten one.
function flatten(m::JuMP.Model, x::VarOrAff)
    return x[:]
end
