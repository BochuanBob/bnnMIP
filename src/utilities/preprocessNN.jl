export preprocNN
function preprocNN(nn)
    nnLen = length(nn)
    inputSize = nn[1]["inputSize"]
    N = length(inputSize)
    nn[1]["inputSize"] = NTuple{N, Int}(inputSize)
    for i in 1:nnLen
        if (haskey(nn[i], "weights") && ~haskey(nn[i], "bias"))
            weights = nn[i]["weights"]
            nn[i]["bias"] = zeros(size(weights)[1])
        elseif (haskey(nn[i], "bias"))
            nn[i]["bias"] = nn[i]["bias"][:]
        end
    end
    # TODO: Work on this function later.
    return nn
end
