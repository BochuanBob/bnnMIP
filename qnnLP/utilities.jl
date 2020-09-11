export readDataset, readOneVar
export readNN

# Read the information of the neural network and store it in desired format.
function readNN(fileName::String, nnName::String)
    file = matopen(fileName)
    nn = read(file, nnName)
    close(file)
    print(nn[1])
    @assert haskey(nn[1], "inputSize")
    inputSize = nn[1]["inputSize"]
    N = length(inputSize)
    nn[1]["inputSize"] = NTuple{N, Int}(inputSize)
    @assert nn[1]["inputSize"] isa NTuple{N, Int} where {N}
    nn[1]["upper"] = ones(nn[1]["inputSize"])
    nn[1]["lower"] = zeros(nn[1]["inputSize"])
    return nn
end


# Read the dataset used to train the neural network.
function readDataset(fileName::String, trainVarName::String,
                    trainLabelName::String,
                    testVarName::String, testLabelName::String)
    file = matopen(fileName)
    trainVar = read(file, trainVarName)
    trainLabel = read(file, trainLabelName)
    testVar = read(file, testVarName)
    testLabel = read(file, testLabelName)
    close(file)
    return trainVar, trainLabel, testVar, testLabel
end

# Read a single variable in the .mat file.
function readOneVar(fileName::String, varName::String)
    file = matopen(fileName)
    var = read(file, varName)
    close(file)
    return var
end
