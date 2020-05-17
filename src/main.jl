using JuMP, Gurobi
include("utilities.jl")
include("verification.jl")
nn = readNN("../data/nn.mat", "nn")
testImages = readOneVar("../data/data.mat", "test_images")
testLabels = readOneVar("../data/data.mat", "test_labels")
testLabels = testLabels[:]
m = direct_model(Gurobi.Optimizer(OutputFlag=1))

i = 1
input=testImages[i,:,:,:]
trueIndex=Int64(testLabels[i])
targetIndex=mod(trueIndex + 1, 10) + 1
epsilon = 0.01
y = perturbationVerify(m, nn, input,trueIndex, targetIndex,epsilon, cuts=false)
optimize!(m)
println(value.(y))
