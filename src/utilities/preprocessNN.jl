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
    nnOut = getNNLayerArray(nn)
    return nnOut
end

function getNNLayerArray(nn)
    nnLen = length(nn)
    nnOut = Array{NNLayer, 1}(undef, nnLen)
    for i in 1:nnLen
        if (nn[i]["type"] == "flatten")
            nnOut[i] = FlattenLayer()
            if (i == 1)
                nnOut[i].inputSize = nn[1]["inputSize"]
            end
        elseif (nn[i]["type"]  == "dense")
            nnOut[i] = DenseLayer()
            nnOut[i].weights = nn[i]["weights"]
            nnOut[i].bias = nn[i]["bias"]
            nnOut[i].upper = nn[i]["upper"]
            nnOut[i].lower = nn[i]["lower"]
            if (haskey(nn[i], "activation"))
                nnOut[i].activation = nn[i]["activation"]
            end
        elseif (nn[i]["type"]  == "denseBin")
            nnOut[i] = DenseBinLayer()
            nnOut[i].weights = nn[i]["weights"]
            nnOut[i].bias = nn[i]["bias"]
            nnOut[i].upper = ones(size(nnOut[i].weights, 2))
            nnOut[i].lower = -ones(size(nnOut[i].weights, 2))
            if (haskey(nn[i], "activation"))
                nnOut[i].activation = nn[i]["activation"]
            end
        elseif (nn[i]["type"]  == "conv2dSign")
            nnOut[i] = Conv2dLayer()
            nnOut[i].weights = nn[i]["weights"]
            nnOut[i].bias = nn[i]["bias"]
            nnOut[i].upper = nn[i]["upper"]
            nnOut[i].lower = nn[i]["lower"]
            nnOut[i].inputSize = nn[i]["inputSize"]
            nnOut[i].padding = nn[i]["padding"]
            nnOut[i].strides = Tuple{Int64, Int64}(nn[i]["strides"])
            # Only support Sign() at this point
            nnOut[i].activation = "Sign"
        elseif (nn[i]["type"]  == "conv2dBinSign")
            nnOut[i] = Conv2dBinLayer()
            nnOut[i].weights = nn[i]["weights"]
            nnOut[i].bias = nn[i]["bias"]
            nnOut[i].padding = nn[i]["padding"]
            nnOut[i].strides = Tuple{Int64, Int64}(nn[i]["strides"])
            # Only support Sign() at this point
            nnOut[i].activation = "Sign"
        else
            error("Not supported layer!")
        end
    end
    return nnOut
end
