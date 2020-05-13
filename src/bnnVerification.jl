using JuMP, Gurobi

include("neuralNetworks.jl")
include("utilities.jl")

m = direct_model(Gurobi.Optimizer(OutputFlag=1))
nn = readNN("nn.mat", "nn")
testImages = readOneVar("data.mat", "test_images")
testLabels = readOneVar("data.mat", "test_labels")
inputSize = nn[1]["inputSize"]
# inputSize = NTuple{3, Int}(inputSize)
@assert inputSize isa NTuple{N, Int} where {N}
x = Array{VariableRef}(undef, inputSize)
for idx in eachindex(x)
    x[idx] = @variable(m)
end
#println(size(testImages[1, :, :, :]))
@constraint(m, x .== testImages[1, :, :, :])
y = getBNNoutput(m, nn, x, cuts=false)
@objective(m, Min, 0)
optimize!(m)
# Do the forward propagation to test the correctness.
println(value.(y))


# Test the input data.
image = testImages[1, :, :, :]
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
println(l3)
