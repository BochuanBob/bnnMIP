export flatten

# Take the input variables and ouput a flatten one.
# Same as the numpy reshape in Python.
# Order: C is the C-like index order, F is the Fortran-like index order.
function flatten(m::JuMP.Model, x::Array{VariableRef},
                upper::Array{Float64}, lower::Array{Float64}; order="C")
    if (order == "C")
        dim = length(size(x))
        xOut = permutedims(x, Array{Int64,1}(dim:-1:1))[:]
        upperOut = permutedims(upper, Array{Int64,1}(dim:-1:1))[:]
        lowerOut = permutedims(lower, Array{Int64,1}(dim:-1:1))[:]
        return xOut, upperOut, lowerOut
    elseif (order == "F")
        return x[:], upper[:], lower[:]
    end
    error("Not supported flatten function!")
    return nothing
end
