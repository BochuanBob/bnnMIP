using JuMP, Gurobi
using LinearAlgebra
include("testFunc.jl")


include("../src/neuralNetworks.jl")
include("../src/utilities.jl")
nn = readNN("../data/nn.mat", "nn")
testImages = readOneVar("../data/data.mat", "test_images")
testLabels = readOneVar("../data/data.mat", "test_labels")
inputSize = nn[1]["inputSize"]
@assert inputSize isa NTuple{N, Int} where {N}

for i in 1:size(testImages, 1)
    m = direct_model(Gurobi.Optimizer(OutputFlag=0))

    x = Array{VariableRef}(undef, inputSize)
    for idx in eachindex(x)
        x[idx] = @variable(m)
    end
    @constraint(m, x .== testImages[i, :, :, :])
    y = getBNNoutput(m, nn, x, cuts=true)
    # Do the forward propagation to test the correctness.

    # Test the input data.
    image = testImages[i, :, :, :]
    l3 = forwardProp(image)
    @objective(m, Min, 0)
    optimize!(m)
    if (mod(i, 10) == 0)
        println("Testing image ", i)
    end
    @assert norm(value.(y) - l3) < 10^(-6)
end
println("Passed all the forward propagation tests.")
