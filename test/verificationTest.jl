using JuMP, Gurobi
using Random
using CSV
using DataFrames

include("../src/utilities.jl")
include("../src/verification.jl")
include("../test/testFunc.jl")
# Inputs
nn = readNN("../data/nn2x100.mat", "nn")
testImages = readOneVar("../data/data.mat", "test_images")
testLabels = readOneVar("../data/data.mat", "test_labels")
testLabels = Array{Int64, 1}(testLabels[:]) .+ 1
num = 10
epsilonList = [0.01]

Random.seed!(1000)
sampleIndex = rand(1:length(testLabels), num)
trueIndices = testLabels[sampleIndex]
targetIndices = Array{Int64, 1}(zeros(num))
for i in 1:num
    targetIndices[i] = rand(setdiff(1:10, trueIndices[i]), 1)[1]
end
timeLimit = 100


for epsilon in epsilonList
    for method in ["AllCuts"]
        for i in [3]
            cuts = false
            if (method == "NoCuts")
                m = direct_model(Gurobi.Optimizer(OutputFlag=1, Cuts=0,
                                TimeLimit=timeLimit, FeasibilityTol=1e-9))
                cuts = false
            elseif (method == "DefaultCuts")
                m = direct_model(Gurobi.Optimizer(OutputFlag=1, Cuts=1,
                                TimeLimit=timeLimit, FeasibilityTol=1e-9))
                cuts = false
            elseif (method == "AllCuts")
                m = direct_model(Gurobi.Optimizer(OutputFlag=1, Cuts=0,
                                TimeLimit=timeLimit, FeasibilityTol=1e-9,
                                IntFeasTol=1e-9))
                set_optimizer_attribute(m, "CutPasses", 100)
                cuts = true
            end
            input=testImages[sampleIndex[i],:,:,:]
            trueIndex=trueIndices[i]
            targetIndex=targetIndices[i]
            x, xInt, y = perturbationVerify(m, nn, input, trueIndex,
                                    targetIndex, epsilon, cuts=cuts, image=true)
            println("Method: ", method)
            println("Epsilon: ", epsilon)
            optimize!(m)
            println("Sample index: ", sampleIndex[i])
            println("True index: ", trueIndex)
            println("Target index: ", targetIndex)
            println("L-infinity norm: ", maximum(abs.(value.(x) - input) ))
            println("Expected Output Based on Input: ",
                forwardProp(value.(x)))
            println("Output: ", value.(y))
            println("Runtime: ", MOI.get(m, Gurobi.ModelAttribute("Runtime")))
            println("Obj: ", MOI.get(m, Gurobi.ModelAttribute("ObjVal")))
            println("Bound: ", MOI.get(m, Gurobi.ModelAttribute("ObjBoundC")))
            println("Node: ", MOI.get(m, Gurobi.ModelAttribute("NodeCount")))
            println("Cons: ", MOI.get(m, Gurobi.ModelAttribute("NumConstrs")))
            println("Iters: ", MOI.get(m, Gurobi.ModelAttribute("IterCount")))
            println("Callback: ", callbackTimeTotal)
        end
    end
end
