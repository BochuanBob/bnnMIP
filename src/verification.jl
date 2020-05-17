using JuMP
include("neuralNetworks.jl")
export perturbationVerify
function perturbationVerify(m::JuMP.Model, nn, input::Array,
                        trueIndex::Int64, targetIndex::Int64,
                        epsilon::Float64; cuts=true)
    inputSize = nn[1]["inputSize"]
    x = Array{VariableRef}(undef, inputSize)
    for idx in eachindex(x)
        x[idx] = @variable(m)
    end
    @constraint(m, x .<= input + epsilon)
    @constraint(m, x .>= input - epsilon)
    y = getBNNoutput(m, nn, x, cuts=cuts)
    @objective(m, Min, y[trueIndex] - y[targetIndex])
    return y
end
