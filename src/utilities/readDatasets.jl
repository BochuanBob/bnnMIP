export readDataset
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
