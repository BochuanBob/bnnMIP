include("layers.jl")
export getBNNoutput

# Add constraints such that y = NN(x).
function getBNNoutput(m::JuMP.Model, nn, x::VarOrAff; cuts=true)
    initNN!(m)
    count = m.ext[:NN].count
    m.ext[:NN].count += 1
    nnLen = length(nn)
    xIn = x
    xOut = nothing
    for i in 1:nnLen
        if (nn[i]["type"] == "flatten")
            xOut = flatten(m, xIn)
        elseif (nn[i]["type"] == "activation")
            xOut = activation1D(m, xIn, nn[i]["function"],
                    upper = nn[i]["upper"], lower = nn[i]["lower"])
        elseif (nn[i]["type"] == "denseBin")
            if (haskey(nn[i], "activation"))
                takeSign = (nn[i]["activation"] == "Sign")
            else
                takeSign = false
            end
            xOut = denseBin(m, xIn, nn[i]["weights"], nn[i]["bias"],
                            takeSign=takeSign, cuts=cuts)
        end
        xIn = xOut
    end
    y = @variable(m, [1:length(xIn)], base_name="y_$count")
    return y
end
