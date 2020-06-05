using JuMP, Gurobi
using Random
using CSV
using DataFrames

include("../src/utilities.jl")
include("../src/verification.jl")
include("../test/testFunc.jl")
# Inputs
nn = readNN("../data/nn3x100.mat", "nn")
testImages = readOneVar("../data/data.mat", "test_images")
testLabels = readOneVar("../data/data.mat", "test_labels")
testLabels = Array{Int64, 1}(testLabels[:]) .+ 1
num = 10
epsilonList = [0.02]

Random.seed!(2020)
nn[3]["weights"] = keepOnlyKEntries(nn[3]["weights"], 5)
# nn[3]["weights"] = keepOnlyKEntriesSeq(nn[3]["weights"], 20)
# nn[4]["weights"] = keepOnlyKEntriesSeq(nn[4]["weights"], 20)
# nn[5]["weights"] = keepOnlyKEntries(nn[5]["weights"], 20)
# nn[6]["weights"] = keepOnlyKEntriesSeq(nn[6]["weights"], 20)
println(length(findall(nn[2]["weights"] .!= 0)))
println(length(findall(nn[3]["weights"] .!= 0)))
sampleIndex = rand(1:length(testLabels), num)
trueIndices = testLabels[sampleIndex]
targetIndices = Array{Int64, 1}(zeros(num))
for i in 1:num
    targetIndices[i] = rand(setdiff(1:10, trueIndices[i]), 1)[1]
end
timeLimit = 2000

for epsilon in epsilonList
    for method in ["DefaultCuts"]
        for i in 1
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
                                Cuts=0, TimeLimit=timeLimit, Threads=4))
                # set_optimizer_attribute(m, "CutPasses", 20000)
                # , MIRCuts=1,
                # ModKCuts=1,NetworkCuts=1,ProjImpliedCuts=1,
                # StrongCGCuts=1
                cuts = true
            elseif (method == "UserCuts")
                m = direct_model(Gurobi.Optimizer(OutputFlag=1, Cuts=0,
                                TimeLimit=timeLimit, PreCrush=1))
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
            x, xInt, y = perturbationVerify(m, nn, input, trueIndex,
                                    targetIndex, epsilon, cuts=cuts,
                                    image=true, preCut=preCut)
            println("Method: ", method)
            println("Epsilon: ", epsilon)
            optimize!(m)
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
            # println("Bound: ", MOI.get(m, Gurobi.ModelAttribute("ObjBoundC")))
            # println("Node: ", MOI.get(m, Gurobi.ModelAttribute("NodeCount")))
            # println("Cons: ", MOI.get(m, Gurobi.ModelAttribute("NumConstrs")))
            # println("Iters: ", MOI.get(m, Gurobi.ModelAttribute("IterCount")))
            println("Callback: ", callbackTimeTotal)
            println("User Cuts Submitted: ", m.ext[:CUTS].count)
        end
    end
end
