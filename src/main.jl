using JuMP, Gurobi
include("utilities.jl")
include("verification.jl")
nn = readNN("../data/nn.mat", "nn")
testImages = readOneVar("../data/data.mat", "test_images")
testLabels = readOneVar("../data/data.mat", "test_labels")
testLabels = testLabels[:]
m = direct_model(Gurobi.Optimizer(OutputFlag=1))

for epsilon in [0.1, 0.15, 0.2]
    for cuts in [false, true]
        i = 1
        input=testImages[i,:,:,:]
        trueIndex=Int64(testLabels[i])
        targetIndex=mod(trueIndex + 1, 10) + 1
        x, y = perturbationVerify(m, nn, input, trueIndex,
                                targetIndex, epsilon, cuts=cuts)
        println("Using our cuts: ", cuts)
        println("Epsilon: ", epsilon)
        optimize!(m)
        println("Input: ", value.(x))
        println("Output: ", value.(y))
    end
end
