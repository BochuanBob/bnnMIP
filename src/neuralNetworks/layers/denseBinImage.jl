include("layerSetup.jl")
include("denseSetup.jl")
export denseBinImage, getDenseBinImageCons

# A fully connected layer with sign() or
# without activation function.
# Each entry of weights must be -1, 0, 1.
function denseBinImage(m::JuMP.Model, xhat::VarOrAff,
               weights::Array{T, 2}, bias::Array{U, 1};
               takeSign=false) where{T<:Real, U<:Real}
    if (~checkWeights(weights))
        error("Each entry of weights must be -1, 0, 1.")
    end
    initNN!(m)
    count = m.ext[:NN].count
    m.ext[:NN].count += 1
    (yLen, xLen) = size(weights)
    if (length(bias) != yLen)
        error("The sizes of weights and bias don't match!")
    end
    y = nothing
    if (takeSign)
        bias = bias .* 255
        x = @variable(m, [1:xLen], base_name="x_$count")
        @constraint(m, x .== 255 .* xhat)
        y = @variable(m, [1:yLen],
                    base_name="y_$count")
        z = @variable(m, [1:yLen], binary=true,
                    base_name="z_$count")
        @constraint(m, [i=1:yLen], y[i] == 2 * z[i] - 1)
        for i in 1:yLen
            neuronSign!(m, x, y[i], weights[i, :], bias[i])
        end
    else
        x = @variable(m, [1:xLen], base_name="x_$count")
        @constraint(m, x .== xhat)
        y = @variable(m, [1:yLen],
                    base_name="y_$count")
        @constraint(m, [i=1:yLen], y[i] ==
                    bias[i] + sum(weights[i,j] * xhat[j] for j in 1:xLen))
    end
    return x, y
end

# Checking each entry of weights must be in -1, 0, 1.
function checkWeights(weights::Array{T, 2}) where{T <: Real}
    for weight in weights
        if (~(weight in [-1, 0, 1]))
            return false
        end
    end
    return true
end

# A MIP formulation for a single neuron.
function neuronSign!(m::JuMP.Model, x::VarOrAff, yi::VarOrAff,
                weightVec::Array{T, 1}, b::U
                ) where{T<:Real, U<:Real}
    # initNN!(m)
    oneIndices = findall(weightVec .== 1)
    negOneIndices = findall(weightVec .== -1)
    nonzeroNum = length(oneIndices) + length(negOneIndices)
    if (nonzeroNum == 0)
        if (b >= 0)
            @constraint(m, yi == 1)
        else
            @constraint(m, yi == -1)
        end
        return nothing
    end
    tau, kappa = getTauAndKappa(nonzeroNum, b)
    Iset1 = union(oneIndices, negOneIndices)
    @constraint(m, getBNNCutFirstConGE(m, x, yi, Iset1,
                oneIndices, negOneIndices, tau)>=0)
    @constraint(m, getBNNCutSecondConLE(m, x, yi, Iset1,
                oneIndices, negOneIndices, kappa)<=0)
    Iset2 = Array{Int64, 1}([])
    @constraint(m, getBNNCutFirstConGE(m, x, yi, Iset2,
                    oneIndices, negOneIndices, tau)>=0)
    @constraint(m, getBNNCutSecondConLE(m, x, yi, Iset2,
                    oneIndices, negOneIndices, kappa)<=0)
    return nothing
end

function getDenseBinImageCons(m::JuMP.Model, xIn::VarOrAff,
                        xOut::VarOrAff, weights::Array{T, 2},
                        bias::Array{U, 1}, cb_data) where{T<:Real, U<:Real}
    conList = []
    (yLen, xLen) = size(weights)
    xVal = zeros(length(xIn))
    bias = 255 .* bias
    for j in 1:xLen
        xVal[j] = JuMP.callback_value(cb_data, xIn[j])
    end
    for i in 1:yLen
        weightVec = weights[i, :]
        b = bias[i]
        oneIndices = findall(weightVec .== 1)
        negOneIndices = findall(weightVec .== -1)
        nonzeroNum = length(oneIndices) + length(negOneIndices)
        if (nonzeroNum == 0)
            continue
        end
        tau, kappa = getTauAndKappa(nonzeroNum, b)
        yVal = JuMP.callback_value(cb_data, xOut[i])
        I1, I2 = getCutsIndices(xVal, yVal,oneIndices,negOneIndices)
        con1 = @build_constraint(getBNNCutFirstConGE(m, xIn, xOut[i], I1,
                        oneIndices, negOneIndices, tau)>=0)
        con2 = @build_constraint(getBNNCutSecondConLE(m, xIn, xOut[i], I2,
                        oneIndices, negOneIndices, kappa) <= 0)
        conList = vcat(conList, con1)
        conList = vcat(conList, con2)
    end
    return conList
end

# Output the I^1 and I^2 for two constraints in Proposition 3
function getCutsIndices(xVal::Array{T, 1}, yVal::U,
                        oneIndices::Array{Int64, 1},
                        negOneIndices::Array{Int64, 1}) where {T<:Real, U<:Real}
    I1 = Array{Int64, 1}([])
    I2 = Array{Int64, 1}([])
    for i in oneIndices
        if (xVal[i] < 255 * yVal)
            append!(I1, i)
        elseif (xVal[i] > 255 * yVal)
            append!(I2, i)
        end
    end

    for i in negOneIndices
        if (-xVal[i] < 255 * yVal)
            append!(I1, i)
        elseif (-xVal[i] > 255 * yVal)
            append!(I2, i)
        end
    end

    return I1, I2
end

# Return first constraint with given I, I^+, I^-, tau, kappa as shown
# in Proposition 3.
function getBNNCutFirstConGE(m::JuMP.Model,
                            x::VarOrAff, yi::VarOrAff,
                            Iset::Array{Int64, 1},
                            oneIndices::Array{Int64, 1},
                            negOneIndices::Array{Int64, 1},
                            tau::T) where {T <: Real}
    Ipos = intersect(Iset, oneIndices)
    Ineg = intersect(Iset, negOneIndices)
    lenI = length(Iset)
    nonzeroNum = length(oneIndices) + length(negOneIndices)
    expr = @expression(m, (sum(x[i] for i in Ipos) - sum(x[i] for i in Ineg))
                    - (((255 * (lenI - nonzeroNum) - tau) * (1 + yi) /2)
                    - (1 - yi) * 255 * lenI/2 ) )
    return expr
end

# Return second constraint with given I, I^+, I^-, tau, kappa as shown
# in Proposition 3.
function getBNNCutSecondConLE(m::JuMP.Model,
                            x::VarOrAff, yi::VarOrAff,
                            Iset::Array{Int64, 1},
                            oneIndices::Array{Int64, 1},
                            negOneIndices::Array{Int64, 1},
                            kappa::T) where {T <: Real}
    Ipos = intersect(Iset, oneIndices)
    Ineg = intersect(Iset, negOneIndices)
    lenI = length(Iset)
    nonzeroNum = length(oneIndices) + length(negOneIndices)
    expr = @expression(m, (sum(x[i] for i in Ipos) - sum(x[i] for i in Ineg)) -
                ( ((1 + yi) * 255 * lenI/2)
                - (255 * (lenI - nonzeroNum) + kappa)*(1 - yi)/2) )
    return expr
end
