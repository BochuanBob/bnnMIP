using JuMP
include("neuralNetworks.jl")
export perturbationVerify

# Minimize the difference of outputs between true index and target index.
function perturbationVerify(m::JuMP.Model, nn, input::Array,
                        trueIndex::Int64, targetIndex::Int64,
                        epsilon::Float64; cuts=true, image=true,
                        integer=false, preCut=false)
    # Don't want to change the original data.
    nnCopy = deepcopy(nn)
    inputSize = nn[1]["inputSize"]
    x = Array{VariableRef}(undef, inputSize)
    xInt = Array{VariableRef}(undef, inputSize)
    for idx in eachindex(x)
        if (image)
            x[idx] = @variable(m, upper_bound=1, lower_bound=0)
        else
            x[idx] = @variable(m)
        end
        if (integer)
            xInt[idx] = @variable(m, integer=true, upper_bound=255,lower_bound=0)
            @constraint(m, 255 * x[idx] == xInt[idx])
        end
    end
    dim = length(size(input))
    inputFlatten = permutedims(input, Array{Int64,1}(dim:-1:1))[:]
    nnCopy[2]["upper"] = min.(nnCopy[2]["upper"], inputFlatten .+ epsilon)
    nnCopy[2]["lower"] = max.(nnCopy[2]["lower"], inputFlatten .- epsilon)
    println(nnCopy[2]["upper"] == nn[2]["upper"])
    @constraint(m, x .<= input .+ epsilon)
    @constraint(m, x .>= input .- epsilon)
    y = getBNNoutput(m, nnCopy, x, cuts=cuts, image=image, preCut=preCut)
    @objective(m, Min, y[trueIndex] - y[targetIndex])
    return x, xInt, y
end

# Find the minimum epsilon such that nn(x_1) = targetIndex
# for ||x_0 - x_1||_infty < epsilon. x_0 is the test data.
function targetVerify(m::JuMP.Model, nn, input::Array,
                        targetIndex::Int64; cuts=true, integer=true)
    inputSize = nn[1]["inputSize"]
    x = Array{VariableRef}(undef, inputSize)
    xInt = Array{VariableRef}(undef, inputSize)
    for idx in eachindex(x)
        x[idx] = @variable(m)
        if (integer)
            xInt[idx] = @variable(m, integer=true, upper_bound=255,lower_bound=0)
            @constraint(m, 255 * x[idx] == xInt[idx])
        end
    end
    epsilon = @variable(m, lower_bound=0)
    @constraint(m, x .<= input + epsilon)
    @constraint(m, x .>= input - epsilon)
    y = getBNNoutput(m, nn, x, cuts=cuts)
    yLen= length(y)
    for i in setdiff(1:yLen, [targetIndex])
        @constraint(m, y[targetIndex] >= y[i] + 1)
    end
    @objective(m, Min, epsilon)
    return x, xInt, y
end

# Find the minimal epsilon such that nn(x_0) \neq nn(x_1)
# for ||x_0 - x_1||_infty < epsilon. x_0 is the test data.
function falsePredictVerify(m::JuMP.Model, nn, input::Array, trueIndex::Int64;
                            cuts=true, preCut=true)
    inputSize = nn[1]["inputSize"]
    x = Array{VariableRef}(undef, inputSize)
    for idx in eachindex(x)
        x[idx] = @variable(m)
    end
    epsilon = @variable(m, lower_bound=0)
    @constraint(m, x .<= input + epsilon)
    @constraint(m, x .>= input - epsilon)
    y = getBNNoutput(m, nn, x, cuts=cuts, preCut=preCut)
    yLen= length(y)
    z = @variable(m)
    zz = @variable(m, [1:yLen], binary=true)
    # Big-M formulation. TODO: Might want to change to
    # other formulation.
    M = 400
    @constraint(m, sum(zz[i] for i in 1:yLen) == 1)
    @constraint(m, zz[trueIndex] == 0)
    for i in setdiff(1:yLen, [trueIndex])
        @constraint(m, z >= y[i])
        @constraint(m, z <= y[i] + M * (1 - zz[i]))
    end
    @constraint(m, z >= y[trueIndex]+1)
    @objective(m, Min, epsilon)
    return x, y
end
