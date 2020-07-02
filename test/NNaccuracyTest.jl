using Pkg
Pkg.activate("../")

import bnnMIP
using JuMP, Gurobi, bnnMIP

include("testFunc.jl")

nn = readNN("../data/nn2F500Sparse.mat", "nn")
testImages = readOneVar("../data/data.mat", "test_images")
testLabels = readOneVar("../data/data.mat", "test_labels")
testLabels = testLabels[:]

totalNum = length(testLabels)
predictCorrect = zeros(totalNum)

for i = 1:totalNum
    output = forwardProp(testImages[i,:,:,:], nn)
    predict = findall(output .== maximum(output))[1]
    if (predict == Int64(testLabels[i]) + 1)
        predictCorrect[i] = 1
    end
end

println("Count: ", sum(predictCorrect), " Total: ", totalNum,
            " Accuracy: ", sum(predictCorrect)/totalNum)
