using Pkg
using Random
using CSV
using DataFrames

Pkg.activate("../")

import bnnMIP
using JuMP, Gurobi, bnnMIP

include("../test/testFunc.jl")
# Inputs
nn = readNN("../data/nn5F100Sparse.mat", "nn")
testImages = readOneVar("../data/data.mat", "test_images")
testLabels = readOneVar("../data/data.mat", "test_labels")
testLabels = Array{Int64, 1}(testLabels[:]) .+ 1
num = 20
epsilonList = [1/255]

for i in 2:(length(nn)-1)
    println("Layer ", i, " zero ratios: ", sum(nn[i].weights .== 0) / length(nn[i].weights))
end

return
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
timeLimit = 5

# NoCuts, DefaultCuts, UserCuts, UserCutsCover, CoverCuts, AllCuts,
#         NoCutsForward, DefaultCutsForward, UserCutsForward,
#         UserCutsCoverForward, CoverCutsForward, AllCutsForward
methodList = ["UserCutsForward"]



para = bnnMIPparameters()
para.useDense = false
para.consistDenseBin = true
for epsilon in epsilonList
    for method in methodList
        for i in 1:1
            methodObj = eval(Meta.parse("bnnMIP." * method))
            m = direct_model(Gurobi.Optimizer(OutputFlag=1, PreCrush=1,
                            Cuts=methodObj.allCuts,
                            CoverCuts=methodObj.coverCuts,
                            TimeLimit=timeLimit, Threads=4))
            input=testImages[sampleIndex[i],:,:,:]
            trueIndex=trueIndices[i]
            targetIndex=targetIndices[i]
            x, xInt, y, nnCopy = perturbationVerify(m, nn, input, trueIndex,
                                    targetIndex, epsilon, para=para,
                                    cuts=methodObj.userCuts,
                                    preCut=methodObj.preCuts,
                                    forward=methodObj.forward)
            println("Method: ", method)
            println("Epsilon: ", epsilon)
            optimize!(m)
            println("Sample index: ", sampleIndex[i])
            println("True index: ", trueIndex)
            println("Target index: ", targetIndex)
            println("L-infinity norm: ", sum(abs.(value.(x) - input)) / length(x) )
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
