using Pkg
using Random
using CSV
using DataFrames

include("qnnLP.jl")

modelList = ["nn2F500DoReFaW1A2.mat", "nn2F500DoReFaW1A2.mat", "nn2F500DoReFaW1A3.mat",
            "nn2F500DoReFaW1A4.mat", "nn2F500DoReFaW1A5.mat", "nn2F500DoReFaW1A6.mat",
            "nn2F500DoReFaW1A7.mat", "nn2F500DoReFaW1A8.mat"]
num = 1
epsilonList = [0.08]
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
timeLimit = 100

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
                                    targetIndex, epsilon)
            println("Model: ", model)
            println("Epsilon: ", epsilon)
            optimize!(m)
            println("Sample index: ", sampleIndex[i])
            println("True index: ", trueIndex)
            println("Target index: ", targetIndex)
            println("L-1 norm: ", sum(abs.(value.(x) - input)) / length(x) )
            println("L-infinity norm: ", maximum(abs.(value.(x) - input)) )
            println("Greater than 1: ", findall(value.(x) .> 1))
            println("Less than 0: ", findall(value.(x) .< 0))
            println("Output: ", value.(y))
            println("Runtime: ", MOI.get(m, Gurobi.ModelAttribute("Runtime")))
            println("Obj: ", MOI.get(m, Gurobi.ModelAttribute("ObjVal")))
            println("Bound: ", MOI.get(m, Gurobi.ModelAttribute("ObjBoundC")))
            println("Cons: ", MOI.get(m, Gurobi.ModelAttribute("NumConstrs")))
            println("Iters: ", MOI.get(m, Gurobi.ModelAttribute("IterCount")))
        end
    end
end
