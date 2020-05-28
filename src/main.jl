using JuMP, Gurobi
using Random
using CSV
using DataFrames

include("utilities.jl")
include("verification.jl")
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
timeLimit = 1000
methodList = ["UserCuts"]

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

count = [1]

for epsilon in epsilonList
    for method in methodList
        for i in 1:num
            cuts = false
            if (method == "NoCuts")
                m = direct_model(Gurobi.Optimizer(OutputFlag=1, Cuts=0,
                                TimeLimit=timeLimit))
                cuts = false
            elseif (method == "DefaultCuts")
                m = direct_model(Gurobi.Optimizer(OutputFlag=1, Cuts=1,
                                TimeLimit=timeLimit))
                cuts = false
            elseif (method == "AllCuts")
                m = direct_model(Gurobi.Optimizer(OutputFlag=1, Cuts=1,
                                TimeLimit=timeLimit, PreCrush=1))
                cuts = true
            elseif (method == "UserCuts")
                m = direct_model(Gurobi.Optimizer(OutputFlag=1, Cuts=0,
                                TimeLimit=timeLimit, PreCrush=1))
                cuts = true
            else
                error("Not supported method.")
            end
            input=testImages[sampleIndex[i],:,:,:]
            trueIndex=trueIndices[i]
            targetIndex=targetIndices[i]
            x, xInt, y = perturbationVerify(m, nn, input, trueIndex,
                                    targetIndex, epsilon, cuts=cuts,
                                    image=true, integer=false)
            println("Method: ", method)
            println("Epsilon: ", epsilon)
            optimize!(m)
            println("L-infinity norm: ", maximum(abs.(value.(x) - input) ))
            println("Expected Output Based on Input: ",
                forwardProp(value.(x)))
            println("Output: ", value.(y))

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
            callbackOut[count[1]] = callbackTimeTotal
            count[1] = count[1] + 1
        end
    end
end

df = DataFrame(Samples=sampleIndexList, TrueIndices=trueIndexList,
            TargetIndices=targetIndexList, Methods=methodsOut,
            Epsilons=epsilonOut, RunTimes=runTimeOut, Objs=objsOut,
            Bounds=boundsOut, NodeCount=nodesOut, NumConstrs=consOut,
            IterCount=itersOut, callbackTimes=callbackOut)
CSV.write("results2x100.csv", df)
