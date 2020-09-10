export perturbationVerify

# Only support dense layers.
# Minimize the difference of outputs between true index and target index.
function perturbationVerify(m::JuMP.Model, nn, input::Array,
                        trueIndex::Int64, targetIndex::Int64,
                        epsilon::Float64)
    # Don't want to change the original data.
    inputSize = nn[1].inputSize
    nnCopy = deepcopy(nn)
    x = Array{VariableRef}(undef, inputSize)
    for idx in eachindex(x)
        x[idx] = @variable(m, upper_bound=1, lower_bound=0)
    end
    nnCopy[1]["upper"] = min.(nnCopy[1].upper, input .+ epsilon)
    nnCopy[1]["lower"] = max.(nnCopy[1].lower, input .- epsilon)

    @constraint(m, x .<= input .+ epsilon)
    @constraint(m, x .>= input .- epsilon)
    y = getDenseQNNoutput(m, nnCopy, x)
    @objective(m, Min, y[trueIndex] - y[targetIndex])
    return x, y
end

function getDenseQNNoutput(m::JuMP.Model, nn,
                    x::Union{VarOrAff, Array{VarOrAff}})
    nnLen = length(nn)
    xCurrent = x
    upper, lower = Float64.(nn[1]["upper"]), Float64.(nn[1]["lower"])
    for i in 1:nnLen
        if nn[i]["type"] in ["denseBin", "dense"]
            xCurrent, upper, lower = denseLP(m, xCurrent, nn[i]["weights"],
                                nn[i]["bias"], upper, lower,
                                nn[i]["activation"])
        elseif nn[i]["type"] == "flatten"
            xCurrent, upper, lower = flatten(m, xCurrent, upper, lower)
        else
            error("Do not support this type of layer!")
        end
    end
    return xOut
end
