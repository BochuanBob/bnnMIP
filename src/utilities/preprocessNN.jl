export preprocNN
function preprocNN(nn)
    nnLen = length(nn)
    inputSize = nn[1]["inputSize"]
    N = length(inputSize)
    nn[1]["inputSize"] = NTuple{N, Int}(inputSize)
    @assert nn[1]["inputSize"] isa NTuple{N, Int} where {N}
    for i in 1:nnLen
        if (haskey(nn[i], "weights"))
            nn[i]["weights"] = Float64.(nn[i]["weights"])
        end
        if (haskey(nn[i], "weights") && ~haskey(nn[i], "bias"))
            weights = nn[i]["weights"]
            nn[i]["bias"] = zeros(size(weights)[1])
        elseif (haskey(nn[i], "bias"))
            if (size(nn[i]["bias"]) == ())
                nn[i]["bias"] = [Float64(nn[i]["bias"])]
            else
                nn[i]["bias"] = Float64.(nn[i]["bias"][:])
            end
        end
        if (haskey(nn[i], "upper") && nn[i]["type"] == "dense")
            nn[i]["upper"] = Float64.(nn[i]["upper"][:])
        end
        if (haskey(nn[i], "lower") && nn[i]["type"] == "dense")
            nn[i]["lower"] = Float64.(nn[i]["lower"][:])
        end
    end
    # TODO: Work on this function later.
    return nn
end
