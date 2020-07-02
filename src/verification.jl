export perturbationVerify

# Minimize the difference of outputs between true index and target index.
function perturbationVerify(m::JuMP.Model, nn::Array{NNLayer, 1}, input::Array,
                        trueIndex::Int64, targetIndex::Int64,
                        epsilon::Float64; para=bnnMIPparameters(),
                        cuts=true, preCut=false, forward=true)
    # Don't want to change the original data.
    integer = para.integer
    image = para.image
    nnCopy = deepcopy(nn)
    inputSize = nn[1].inputSize
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
    for i in 1:length(nnCopy)
        if (typeof(nnCopy[i]) == DenseLayer)
            dim = length(size(input))
            inputFlatten = permutedims(input, Array{Int64,1}(dim:-1:1))[:]
            nnCopy[i].upper = min.(nnCopy[i].upper, inputFlatten .+ epsilon)
            nnCopy[i].lower = max.(nnCopy[i].lower, inputFlatten .- epsilon)
            break
        elseif (typeof(nnCopy[i]) == Conv2dLayer)
            nnCopy[i].upper = min.(nnCopy[i].upper, input .+ epsilon)
            nnCopy[i].lower = max.(nnCopy[i].lower, input .- epsilon)
            break
        end
    end
    @constraint(m, x .<= input .+ epsilon)
    @constraint(m, x .>= input .- epsilon)
    y, nnCopy = getBNNoutput(m, nnCopy, x, para, cuts=cuts,
                    preCut=preCut, forward=forward)
    @objective(m, Min, y[trueIndex] - y[targetIndex])
    return x, xInt, y, nnCopy
end


# # Find the minimum epsilon such that nn(x_1) = targetIndex
# # for ||x_0 - x_1||_infty < epsilon. x_0 is the test data.
# # TODO: Need to modify later.
# function targetVerify(m::JuMP.Model, nn, input::Array,
#                         targetIndex::Int64; cuts=true, integer=true)
#     inputSize = nn[1]["inputSize"]
#     x = Array{VariableRef}(undef, inputSize)
#     xInt = Array{VariableRef}(undef, inputSize)
#     for idx in eachindex(x)
#         x[idx] = @variable(m)
#         if (integer)
#             xInt[idx] = @variable(m, integer=true, upper_bound=255,lower_bound=0)
#             @constraint(m, 255 * x[idx] == xInt[idx])
#         end
#     end
#     epsilon = @variable(m, lower_bound=0)
#     @constraint(m, x .<= input + epsilon)
#     @constraint(m, x .>= input - epsilon)
#     y = getBNNoutput(m, nn, x, cuts=cuts)
#     yLen= length(y)
#     for i in setdiff(1:yLen, [targetIndex])
#         @constraint(m, y[targetIndex] >= y[i] + 1)
#     end
#     @objective(m, Min, epsilon)
#     return x, xInt, y
# end
#
#
# # Find the minimal epsilon such that nn(x_0) \neq nn(x_1)
# # for ||x_0 - x_1||_infty < epsilon. x_0 is the test data.
# # TODO: Need to modify later.
# function falsePredictVerify(m::JuMP.Model, nn, input::Array, trueIndex::Int64;
#                             cuts=true, preCut=true)
#     inputSize = nn[1]["inputSize"]
#     x = Array{VariableRef}(undef, inputSize)
#     for idx in eachindex(x)
#         x[idx] = @variable(m)
#     end
#     epsilon = @variable(m, lower_bound=0)
#     @constraint(m, x .<= input + epsilon)
#     @constraint(m, x .>= input - epsilon)
#     y = getBNNoutput(m, nn, x, cuts=cuts, preCut=preCut)
#     yLen= length(y)
#     z = @variable(m)
#     zz = @variable(m, [1:yLen], binary=true)
#     # Big-M formulation. TODO: Might want to change to
#     # other formulation.
#     M = 400
#     @constraint(m, sum(zz[i] for i in 1:yLen) == 1)
#     @constraint(m, zz[trueIndex] == 0)
#     for i in setdiff(1:yLen, [trueIndex])
#         @constraint(m, z >= y[i])
#         @constraint(m, z <= y[i] + M * (1 - zz[i]))
#     end
#     @constraint(m, z >= y[trueIndex]+1)
#     @objective(m, Min, epsilon)
#     return x, y
# end
