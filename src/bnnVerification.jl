using JuMP, Gurobi

include("neuralNetworks.jl")
include("utilities.jl")

m = direct_model(Gurobi.Optimizer(OutputFlag=1))
nn = readNN("nn.mat", "nn")
testImages = readFromMAT("data.mat", "test_images")
testLabels = readFromMAT("data.mat", "test_labels")
inputSize = nn[1]["inputSize"]
@assert inputSize isa NTuple{N, Int} where {N}
x = Array{VariableRef}(undef, inputSize)
for idx in eachindex(x)
    x[idx] = @variable(m)
end
@constraint(m, x .== testImages[1, :, :, :])
y = getBNNoutput(m, nn, x)
@objective(m, Min, 0)
optimize!(m)
# Do the forward propagation to test the correctness.
println(value.(y))
