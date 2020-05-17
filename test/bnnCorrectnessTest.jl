using JuMP, Gurobi

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
    image = permutedims(image, [3,2,1])[:]
    weight1 = nn[2]["weights"]
    bias1 = nn[2]["bias"]
    weight2 = nn[3]["weights"]
    bias2 = nn[3]["bias"]
    weight3 = nn[4]["weights"]
    bias3 = nn[4]["bias"]

    l1 = sign.(weight1 * image + bias1)
    l2 = sign.(weight2 * l1 + bias2)
    l3 = weight3 * l2 + bias3

    # @constraint(m, y .== l3)
    @objective(m, Min, 0)
    optimize!(m)
    if (mod(i, 10) == 0)
        println("Testing image ", i)
    end
    @assert value.(y) == l3
end
println("Passed all the forward propagation tests.")
