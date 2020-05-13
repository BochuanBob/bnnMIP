include("layerSetup.jl")
export flatten

# Take the input variables and ouput a flatten one.
# Same as the numpy reshape in Python.
# Order: C is the C-like index order, F is the Fortran-like index order.
function flatten(m::JuMP.Model, x::VarOrAff; order="C")
    if (order == "C")
        dim = length(size(x))
        return permutedims(x, Array{Int64,1}(dim:-1:1))[:]
    elseif (order = "F")
        return x[:]
    end
    error("Not supported flatten function!")
    return nothing
end
