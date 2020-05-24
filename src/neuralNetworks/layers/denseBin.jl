include("layerSetup.jl")
include("denseSetup.jl")
export denseBin, addDenseBinCons!

# A fully connected layer with sign() or
# without activation function.
# Each entry of weights must be -1, 0, 1.
function denseBin(m::JuMP.Model, x::VarOrAff,
               weights::Array{T, 2}, bias::Array{U, 1};
               takeSign=false, image=true) where{T<:Real, U<:Real}
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
    tauList = zeros(yLen)
    kappaList = zeros(yLen)
    oneIndicesList = Array{Array{Int64, 1}, 1}(undef, yLen)
    negOneIndicesList = Array{Array{Int64, 1}, 1}(undef, yLen)
    y = nothing
    if (takeSign)
        y = @variable(m, [1:yLen],
                    base_name="y_$count")
        z = @variable(m, [1:yLen], binary=true,
                    base_name="z_$count")
        @constraint(m, [i=1:yLen], y[i] == 2 * z[i] - 1)
        for i in 1:yLen
            tauList[i], kappaList[i], oneIndicesList[i], negOneIndicesList[i] =
                neuronSign(m, x, y[i], weights[i, :], bias[i], image=image)
        end
    else
        y = @variable(m, [1:yLen],
                    base_name="y_$count")
        @constraint(m, [i=1:yLen], y[i] ==
                    bias[i] + sum(weights[i,j] * x[j] for j in 1:xLen))
    end
    return y, tauList, kappaList, oneIndicesList, negOneIndicesList
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
function neuronSign(m::JuMP.Model, x::VarOrAff, yi::VarOrAff,
                weightVec::Array{T, 1}, b::U;
                image=true) where{T<:Real, U<:Real}
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
    if (image)
        tau, kappa = getTauAndKappa(nonzeroNum, b)
    else
        tau, kappa = b, b
    end
    Iset1 = union(oneIndices, negOneIndices)
    @constraint(m, getBNNCutFirstConGE(m, x, yi, Iset1,
                oneIndices, negOneIndices, tau)>=0)
    @constraint(m, getBNNCutSecondConLE(m, x, yi, Iset1,
                oneIndices, negOneIndices, kappa)<=0)
    return tau, kappa, oneIndices, negOneIndices
end

function addDenseBinCons!(m::JuMP.Model, xIn::VarOrAff,
                        xOut::VarOrAff,tauList::Array{Float64, 1},
                        kappaList::Array{Float64, 1},
                        oneIndicesList::Array{Array{Int64, 1}, 1},
                        negOneIndicesList::Array{Array{Int64, 1}, 1}, cb_data)
    yLen, xLen = length(xOut), length(xIn)
    xVal = zeros(xLen)
    for j in 1:xLen
        xVal[j] = JuMP.callback_value(cb_data, xIn[j])
    end
    for i in 1:yLen
        yVal = JuMP.callback_value(cb_data, xOut[i])
        if (abs(yVal - 1) < 10^(-4) || abs(yVal + 1) < 10^(-4))
            continue
        end
        oneIndices = oneIndicesList[i]
        negOneIndices = negOneIndicesList[i]
        nonzeroNum = length(oneIndices) + length(negOneIndices)
        if (nonzeroNum == 0)
            continue
        end
        tau, kappa = tauList[i], kappaList[i]
        I1pos, I1neg, I2pos, I2neg =
                    getBinCutsIndices(xVal, yVal,oneIndices,negOneIndices)
        lenI1 = length(I1pos) + length(I1neg)
        lenI2 = length(I2pos) + length(I2neg)
        con1 = getfirstBinCon(xIn,xOut[i],I1pos,I1neg,lenI1,nonzeroNum,tau)
        con2 = getSecondBinCon(xIn,xOut[i],I2pos,I2neg,lenI2,nonzeroNum,kappa)
        MOI.submit(m, MOI.UserCut(cb_data), con1)
        MOI.submit(m, MOI.UserCut(cb_data), con2)
    end
    return nothing
end

# Output the I^1 and I^2 for two constraints in Proposition 3
function getBinCutsIndices(xVal::Array{T, 1}, yVal::U,
                        oneIndices::Array{Int64, 1},
                        negOneIndices::Array{Int64, 1}) where {T<:Real, U<:Real}
    I1pos = Array{Int64, 1}([])
    I1neg = Array{Int64, 1}([])
    I2pos = Array{Int64, 1}([])
    I2neg = Array{Int64, 1}([])
    for i in oneIndices
        if (xVal[i] < yVal)
            append!(I1pos, i)
        elseif (xVal[i] > yVal)
            append!(I2pos, i)
        end
    end

    for i in negOneIndices
        if (-xVal[i] < yVal)
            append!(I1neg, i)
        elseif (-xVal[i] > yVal)
            append!(I2neg, i)
        end
    end

    return I1pos, I1neg, I2pos, I2neg
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
                    -(((lenI - nonzeroNum - tau) * (1 + yi) /2) - (1 - yi) * lenI/2 ) )
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
                ( ((1 + yi) * lenI/2) - (lenI - nonzeroNum + kappa)*(1 - yi)/2))
    return expr
end


function getfirstBinCon(x::VarOrAff, yi::VarOrAff,
                            Ipos::Array{Int64, 1},
                            Ineg::Array{Int64, 1},
                            lenI::Int64,
                            nonzeroNum::Int64,
                            tau::T) where {T <: Real}
    return @build_constraint((sum(x[i] for i in Ipos) - sum(x[i] for i in Ineg))
                    >=(((lenI - nonzeroNum - tau) * (1 + yi) /2) - (1 - yi) * lenI/2 ))
end

# Return second constraint with given I, I^+, I^-, tau, kappa as shown
# in Proposition 3.
function getSecondBinCon(x::VarOrAff, yi::VarOrAff,
                            Ipos::Array{Int64, 1},
                            Ineg::Array{Int64, 1},
                            lenI::Int64,
                            nonzeroNum::Int64,
                            kappa::T) where {T <: Real}
    return @build_constraint((sum(x[i] for i in Ipos) - sum(x[i] for i in Ineg))
                <=(((1 + yi) * lenI/2) - (lenI - nonzeroNum + kappa)*(1 - yi)/2))
end
