using Pkg
using Random
using CSV
using DataFrames

include("qnnLP.jl")

modelList = ["nn2F100DoReFaW1A2.mat",
            "nn2F100DoReFaW1A4.mat", "nn2F100DoReFaW1A8.mat",
            "nn2F100DoReFaW1A16.mat", "nn2F100DoReFaW1A32.mat"]
MIP = false
num = 10
epsilonList = [0.05]
Random.seed!(2020)
testImages = readOneVar("../data/data.mat", "test_images")
testLabels = readOneVar("../data/data.mat", "test_labels")
testLabels = Array{Int64, 1}(testLabels[:]) .+ 1
sampleIndex = rand(1:length(testLabels), num)
trueIndices = testLabels[sampleIndex]
targetIndices = Array{Int64, 1}(zeros(num))

for i in 1:num
    targetIndices[i] = rand(setdiff(1:10, trueIndices[i]), 1)[1]
end
timeLimit = 50
totalLen = length(modelList) * num * length(epsilonList)
sampleIndexList = Array{Int64, 1}(zeros(totalLen))
trueIndexList = Array{Int64, 1}(zeros(totalLen))
targetIndexList = Array{Int64, 1}(zeros(totalLen))
modelsOut = Array{String, 1}(undef, totalLen)
epsilonOut = zeros(totalLen)

runTimeOut = zeros(totalLen)
objsOut = zeros(totalLen)
if (MIP)
    boundsOut = zeros(totalLen)
end
consOut = zeros(totalLen)
itersOut = zeros(totalLen)
InstanceOut = zeros(totalLen)
count = [1]

for model in modelList
    println(model)
    nn = readNN(string("../data/", model), "nn")
    for epsilon in epsilonList
        for i in 1:num
            println("-" ^ (length("Instance: ") + 2))
            println("Instance: ", i)
            println("-" ^ (length("Instance: ") + 2))
            m = direct_model(Gurobi.Optimizer(OutputFlag=1,
                            TimeLimit=timeLimit, Threads=4))
            input=testImages[sampleIndex[i],:,:,:]
            trueIndex=trueIndices[i]
            targetIndex=targetIndices[i]
            x, y = perturbationVerify(m, nn, input, trueIndex,
                                    targetIndex, epsilon, MIP=MIP)
            println("Model: ", model)
            println("Epsilon: ", epsilon)
            optimize!(m)
            # println("Sample index: ", sampleIndex[i])
            # println("True index: ", trueIndex)
            # println("Target index: ", targetIndex)
            # println("L-1 norm: ", sum(abs.(value.(x) - input)) / length(x) )
            # println("L-infinity norm: ", maximum(abs.(value.(x) - input)) )
            # println("Greater than 1: ", findall(value.(x) .> 1))
            # println("Less than 0: ", findall(value.(x) .< 0))
            # println("Output: ", value.(y))
            # println("Runtime: ", MOI.get(m, Gurobi.ModelAttribute("Runtime")))
            println("Obj: ", MOI.get(m, Gurobi.ModelAttribute("ObjVal")))
            if (MIP)
                println("Bound: ", MOI.get(m, Gurobi.ModelAttribute("ObjBoundC")))
            end
            # println("Cons: ", MOI.get(m, Gurobi.ModelAttribute("NumConstrs")))
            # println("Iters: ", MOI.get(m, Gurobi.ModelAttribute("IterCount")))
            InstanceOut[count[1]] = i
            sampleIndexList[count[1]] = sampleIndex[i]
            trueIndexList[count[1]] = trueIndex
            targetIndexList[count[1]] = targetIndex
            modelsOut[count[1]] = model
            epsilonOut[count[1]] = epsilon
            runTimeOut[count[1]] = MOI.get(m, Gurobi.ModelAttribute("Runtime"))
            objsOut[count[1]] = MOI.get(m, Gurobi.ModelAttribute("ObjVal"))
            if (MIP)
                boundsOut[count[1]] = MOI.get(m, Gurobi.ModelAttribute("ObjBoundC"))
            end
            consOut[count[1]] = MOI.get(m, Gurobi.ModelAttribute("NumConstrs"))
            itersOut[count[1]] = MOI.get(m, Gurobi.ModelAttribute("IterCount"))
            count[1] = count[1] + 1
        end
    end
end

if (MIP)
    df = DataFrame(Instance=InstanceOut, Samples=sampleIndexList,
                TrueIndices=trueIndexList,
                TargetIndices=targetIndexList, Models=modelsOut,
                Epsilons=epsilonOut, RunTimes=runTimeOut, Objs=objsOut,
                bounds=boundsOut,
                NumConstrs=consOut, IterCount=itersOut)
else
    df = DataFrame(Instance=InstanceOut, Samples=sampleIndexList,
                TrueIndices=trueIndexList,
                TargetIndices=targetIndexList, Models=modelsOut,
                Epsilons=epsilonOut, RunTimes=runTimeOut, Objs=objsOut,
                NumConstrs=consOut, IterCount=itersOut)
end
CSV.write("qnnOutput/test.csv", df)
