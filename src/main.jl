using Pkg
using Random
using CSV
using DataFrames

Pkg.activate("../")

import bnnMIP
using JuMP, Gurobi, bnnMIP

include("../test/testFunc.jl")
# Inputs
nn = readNN("../data/nn2F500Sparse.mat", "nn")
testImages = readOneVar("../data/data.mat", "test_images")
testLabels = readOneVar("../data/data.mat", "test_labels")
testLabels = Array{Int64, 1}(testLabels[:]) .+ 1
num = 20
epsilonList = [0.08]

Random.seed!(2020)
sampleIndex = rand(1:length(testLabels), num)
trueIndices = testLabels[sampleIndex]
targetIndices = Array{Int64, 1}(zeros(num))
for i in 1:num
    targetIndices[i] = rand(setdiff(1:10, trueIndices[i]), 1)[1]
end
timeLimit = 1800
methodList = ["NoCuts", "UserCuts", "CoverCuts",
                "DefaultCuts", "AllCuts", "UserCutsCover",
                "NoCutsForward", "UserCutsForward", "CoverCutsForward",
                "DefaultCutsForward", "AllCutsForward", "UserCutsCoverForward"]

# Ouputs
totalLen = length(methodList)*length(epsilonList)*num
sampleIndexList = Array{Int64, 1}(zeros(totalLen))
trueIndexList = Array{Int64, 1}(zeros(totalLen))
targetIndexList = Array{Int64, 1}(zeros(totalLen))
methodsOut = Array{String, 1}(undef, totalLen)
epsilonOut = zeros(totalLen)

runTimeOut = zeros(totalLen)
objsOut = zeros(totalLen)
boundsOut = zeros(totalLen)
nodesOut = zeros(totalLen)
consOut = zeros(totalLen)
itersOut = zeros(totalLen)
callbackOut = zeros(totalLen)
userCutsOut = zeros(totalLen)
InstanceOut = zeros(totalLen)

count = [1]
para = bnnMIPparameters()

for epsilon in epsilonList
    for method in methodList
        for i in 1:num
            methodObj = eval(Meta.parse(method * "()"))
            m = direct_model(Gurobi.Optimizer(OutputFlag=1, PreCrush=1,
                            Cuts=methodObj.allCuts,
                            CoverCuts=methodObj.coverCuts,
                            TimeLimit=timeLimit, Threads=4))
            input=testImages[sampleIndex[i],:,:,:]
            trueIndex=trueIndices[i]
            targetIndex=targetIndices[i]
            x, xInt, y, nnCopy = perturbationVerify(m, nn, input, trueIndex,
                                    targetIndex, epsilon,
                                    cuts=methodObj.userCuts,
                                    preCut=methodObj.preCuts,
                                    forward=methodObj.forward)
            println("Method: ", method)
            println("Epsilon: ", epsilon)
            optimize!(m)
            println("L-infinity norm: ", maximum(abs.(value.(x) - input) ))
            println("Expected Output Based on Input: ",
                forwardProp(value.(x), nn))
            println("Output: ", value.(y))

            InstanceOut[count[1]] = i
            sampleIndexList[count[1]] = sampleIndex[i]
            trueIndexList[count[1]] = trueIndex
            targetIndexList[count[1]] = targetIndex
            methodsOut[count[1]] = method
            epsilonOut[count[1]] = epsilon
            runTimeOut[count[1]] = MOI.get(m, Gurobi.ModelAttribute("Runtime"))
            objsOut[count[1]] = MOI.get(m, Gurobi.ModelAttribute("ObjVal"))
            boundsOut[count[1]] = MOI.get(m, Gurobi.ModelAttribute("ObjBoundC"))
            nodesOut[count[1]] = MOI.get(m, Gurobi.ModelAttribute("NodeCount"))
            consOut[count[1]] = MOI.get(m, Gurobi.ModelAttribute("NumConstrs"))
            itersOut[count[1]] = MOI.get(m, Gurobi.ModelAttribute("IterCount"))
            callbackOut[count[1]] = m.ext[:CALLBACK_TIME].time
            userCutsOut[count[1]]= m.ext[:CUTS].count
            count[1] = count[1] + 1
        end
    end
end

df = DataFrame(Instance=InstanceOut, Samples=sampleIndexList,
            TrueIndices=trueIndexList,
            TargetIndices=targetIndexList, Methods=methodsOut,
            Epsilons=epsilonOut, RunTimes=runTimeOut, Objs=objsOut,
            Bounds=boundsOut, NodeCount=nodesOut, NumConstrs=consOut,
            IterCount=itersOut, callbackTimes=callbackOut,
            submittedCuts=userCutsOut)
CSV.write("results2F500Ep008All.csv", df)
