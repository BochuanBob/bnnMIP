using JuMP, Gurobi
using LinearAlgebra
using StatsFuns
include("testFunc.jl")

include("../src/neuralNetworks.jl")
include("../src/utilities.jl")
nn = readNN("../data/nnConv2LayersSame.mat", "nn")
testImages = readOneVar("../data/data.mat", "test_images")
testLabels = readOneVar("../data/data.mat", "test_labels")
inputSize = nn[1]["inputSize"]
println(size(nn[1]["upper"]))
@assert inputSize isa NTuple{N, Int} where {N}

for i in 1:10
    m = direct_model(Gurobi.Optimizer(OutputFlag=0))

    x = Array{VariableRef}(undef, inputSize)
    for idx in eachindex(x)
        x[idx] = @variable(m)
    end
    @constraint(m, x .== testImages[i, :, :, :])
    y = getBNNoutput(m, nn, x, cuts=false, preCut=false)
    # Do the forward propagation to test the correctness.
    @objective(m, Min, 0)
    optimize!(m)
    println("softmax value: ", softmax(value.(y)))
end
