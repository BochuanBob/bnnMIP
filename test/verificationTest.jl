using JuMP
using Random
using CSV
using DataFrames

include("../src/utilities.jl")
include("../src/verification.jl")
include("../test/testFunc.jl")
# Inputs
nn = readNN("../data/nn2x500AllSparseE12.mat", "nn")
testImages = readOneVar("../data/data.mat", "test_images")
testLabels = readOneVar("../data/data.mat", "test_labels")
testLabels = Array{Int64, 1}(testLabels[:]) .+ 1
num = 20
epsilonList = [0.01]

Random.seed!(2020)
# nn[3]["weights"] = keepOnlyKEntriesSeq(nn[3]["weights"], 20)
# nn[4]["weights"] = keepOnlyKEntriesSeq(nn[4]["weights"], 20)
# nn[5]["weights"] = keepOnlyKEntries(nn[5]["weights"], 20)
# nn[6]["weights"] = keepOnlyKEntriesSeq(nn[6]["weights"], 20)
sampleIndex = rand(1:length(testLabels), num)
trueIndices = testLabels[sampleIndex]
targetIndices = Array{Int64, 1}(zeros(num))
for i in 1:num
    targetIndices[i] = rand(setdiff(1:10, trueIndices[i]), 1)[1]
end
timeLimit = 100

for epsilon in epsilonList
    for method in ["UserCuts"]
        for i in 9:9
            cuts = false
            preCut = false
            if (method == "NoCuts")
                m = direct_model(Gurobi.Optimizer(OutputFlag=1, PreCrush=1, Cuts=0,
                                TimeLimit=timeLimit, Threads=4))
            elseif (method == "DefaultCuts")
                m = direct_model(Gurobi.Optimizer(OutputFlag=1, PreCrush=1, Cuts=1,
                                TimeLimit=timeLimit, Threads=4))
            elseif (method == "AllCuts")
                m = direct_model(Gurobi.Optimizer(OutputFlag=1, PreCrush=1,
                                Cuts=1, TimeLimit=timeLimit, Threads=4))
                cuts = true
            elseif (method == "UserCuts")
                m = direct_model(Gurobi.Optimizer(OutputFlag=1, Cuts=0,
                                TimeLimit=timeLimit, PreCrush=1, Threads=4))
                cuts = true
            elseif (method == "PreCut")
                m = direct_model(Gurobi.Optimizer(OutputFlag=1, PreCrush=1, Cuts=0,
                                TimeLimit=timeLimit, Threads=4))
                preCut = true
            elseif (method == "PreCutDefault")
                m = direct_model(Gurobi.Optimizer(OutputFlag=1, PreCrush=1, Cuts=1,
                                TimeLimit=timeLimit, Threads=4))
                preCut = true
            elseif (method == "PreCutAll")
                m = direct_model(Gurobi.Optimizer(OutputFlag=1, PreCrush=1, Cuts=1,
                                TimeLimit=timeLimit, Threads=4))
                preCut = true
                cuts = true
            elseif (method == "PreCutUser")
                m = direct_model(Gurobi.Optimizer(OutputFlag=1, PreCrush=1, Cuts=0,
                                TimeLimit=timeLimit, Threads=4))
                preCut = true
                cuts = true
            end

            input=testImages[sampleIndex[i],:,:,:]
            trueIndex=trueIndices[i]
            targetIndex=targetIndices[i]
            x, xInt, y, nnCopy = perturbationVerify(m, nn, input, trueIndex,
                                    targetIndex, epsilon, cuts=cuts,
                                    image=true, preCut=preCut)
            println("Method: ", method)
            println("Epsilon: ", epsilon)
            myOptimize!(backend(m), nnCopy)
            println("Sample index: ", sampleIndex[i])
            println("True index: ", trueIndex)
            println("Target index: ", targetIndex)
            println("L-infinity norm: ", maximum(abs.(value.(x) - input) ))
            println("Greater than 1: ", findall(value.(x) .> 1))
            println("Less than 0: ", findall(value.(x) .< 0))
            println("Expected Output Based on Input: ",
                forwardProp(value.(x), nn))
            println("Output: ", value.(y))
            println("Runtime: ", MOI.get(m, Gurobi.ModelAttribute("Runtime")))
            println("Obj: ", MOI.get(m, Gurobi.ModelAttribute("ObjVal")))
            println("Bound: ", MOI.get(m, Gurobi.ModelAttribute("ObjBoundC")))
            println("Node: ", MOI.get(m, Gurobi.ModelAttribute("NodeCount")))
            println("Cons: ", MOI.get(m, Gurobi.ModelAttribute("NumConstrs")))
            println("Iters: ", MOI.get(m, Gurobi.ModelAttribute("IterCount")))
            println("Callback: ", m.ext[:CALLBACK_TIME].time)
            println("bench_conv2d: ", m.ext[:BENCH_CONV2D].time)
            println("TEST_CONSTRAINTS: ", m.ext[:TEST_CONSTRAINTS].count)
            println("User Cuts Submitted: ", m.ext[:CUTS].count)
        end
    end
end
