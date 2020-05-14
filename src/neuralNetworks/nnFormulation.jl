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
    xOnes = false
    for i in 1:nnLen
        if (nn[i]["type"] == "flatten")
            xOut = flatten(m, xIn)
        elseif (nn[i]["type"] == "activation")
            xOut = activation1D(m, xIn, nn[i]["function"],
                    upper = nn[i]["upper"], lower = nn[i]["lower"])
        elseif (nn[i]["type"] == "denseBin")
            takeSign = false
            if (haskey(nn[i], "activation"))
                takeSign = (nn[i]["activation"] == "Sign")
            end
            # After the first sign() function, the input of each binary layer
            # has entries of -1 and 1.
            xOnes = xOnes || takeSign
            xOut = denseBin(m, xIn, nn[i]["weights"], nn[i]["bias"],
                            takeSign=takeSign, cuts=cuts, xOnes=xOnes)
        else
            error("Not support layer.")
        end
        xIn = xOut
    end
    y = xOut
    return y
end
