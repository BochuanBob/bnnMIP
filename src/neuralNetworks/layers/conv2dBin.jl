include("layerSetup.jl")
include("denseSetup.jl")
export conv2dBinSign

const CUTOFF_CONV2D_BIN = 10
const CUTOFF_CONV2D_BIN_PRECUT = 5

# A conv2d layer with sign().
# Each entry of weights must be -1, 0, 1.
function conv2dBinSign(m::JuMP.Model, x::VarOrAff,
                weights::Array{T, 4}, bias::Array{U, 1},
                strides::Tuple{Int64, Int64}; padding="valid",
                image=true, preCut=true, cuts=true) where{T<:Real, U<:Real}
    if (~checkWeights(weights))
        error("Each entry of weights must be -1, 0, 1.")
    end
    EMPTY3DVAR = Array{VariableRef, 3}(undef, (0,0,0))
    EMPTY3DFLOAT = Array{Float64, 3}(undef, (0,0,0))
    initNN!(m)
    count = m.ext[:NN].count
    m.ext[:NN].count += 1
    (k1Len, k2Len, channels, filters) = size(weights)
    (s1Len, s2Len) = strides
    (x1Len, x2Len, x3Len) = size(x)

    if (length(bias) != filters)
        error("The sizes of weights and bias don't match!")
    end
    if (x3Len != channels)
        error("The length of channels doesn't match!")
    end
    y3Len = Int64(filters)
    if (padding == "valid")
        y1Len = Int64(floor((x1Len - k1Len) / s1Len) + 1)
        y2Len = Int64(floor((x2Len - k2Len) / s2Len) + 1)
    elseif (padding == "same")
        y1Len = Int64(ceil(x1Len / s1Len))
        y2Len = Int64(ceil(x2Len / s2Len))
    else
        error("Not supported padding!")
    end

    outputSize = (y1Len, y2Len, y3Len)
    tauList = zeros(outputSize)
    kappaList = zeros(outputSize)
    oneIndicesList = Array{Array{CartesianIndex{3}, 1}, 3}(undef, outputSize)
    negOneIndicesList = Array{Array{CartesianIndex{3}, 1}, 3}(undef, outputSize)
    z = @variable(m, [1:y1Len, 1:y2Len, 1:y3Len], binary=true,
                base_name="z_$count")
    y = @expression(m, 2 .* z .- 1)
    for i in 1:y1Len
        for j in 1:y2Len
            for k in 1:y3Len
                weight = EMPTY3DFLOAT
                x1Start, x1End = 1 + (i-1)*s1Len, (i-1)*s1Len + k1Len
                x2Start, x2End = 1 + (j-1)*s2Len, (j-1)*s2Len + k2Len
                xN = EMPTY3DVAR
                if (x1Start <= x1Len && x2Start <= x2Len)
                    xN = x[x1Start:min(x1Len, x1End),
                            x2Start:min(x2Len, x2End), :]
                    (xN1Len, xN2Len, _) = size(xN)
                    weight = weights[1:xN1Len, 1:xN2Len, :, k]
                end
                tauList[i,j,k], kappaList[i,j,k], oneIndicesList[i,j,k],
                    negOneIndicesList[i,j,k] = neuronSign(m, xN, y[i,j,k],
                            weight, bias[k], image=image, preCut=preCut,
                            cuts=cuts)
            end
        end
    end
    return y, tauList, kappaList, oneIndicesList, negOneIndicesList
end

# A MIP formulation for a single neuron.
function neuronSign(m::JuMP.Model, x::VarOrAff, yijk::VarOrAff,
                weightVec::Array{T, 3}, b::U;
                image=true, preCut=true, cuts=true) where{T<:Real, U<:Real}
    # initNN!(m)
    oneIndices = findall(weightVec .== 1)
    negOneIndices = findall(weightVec .== -1)
    nonzeroNum = length(oneIndices) + length(negOneIndices)

    if (nonzeroNum == 0 || size(x) == (0,0,0))
        if (b >= 0)
            @constraint(m, yijk == 1)
        else
            @constraint(m, yijk == -1)
        end
        return b, b, oneIndices, negOneIndices
    end
    # For comparison to not handling discontinuity.
    if (image)
        tau, kappa = getTauAndKappa(nonzeroNum, b)
    else
        tau, kappa = b, b
    end
    if (tau >= nonzeroNum)
        @constraint(m, yijk == 1)
        return tau, kappa, oneIndices, negOneIndices
    end
    if (kappa <= -nonzeroNum)
        @constraint(m, yijk == -1)
        return tau, kappa, oneIndices, negOneIndices
    end
    if (preCut && (nonzeroNum <= CUTOFF_CONV2D_BIN_PRECUT) )
        IsetAll = collect(powerset(union(oneIndices, negOneIndices)))
        IsetAll = IsetAll[2:length(IsetAll)]
        (var, _) = collect(yijk.terms)[1]
        MOI.set(m, Gurobi.VariableAttribute("BranchPriority"), var, -1000)
    else
        IsetAll = [union(oneIndices, negOneIndices)]
    end
    for Iset1 in IsetAll
        Ipos = intersect(Iset1, oneIndices)
        Ineg = intersect(Iset1, negOneIndices)
        lenI = length(Iset1)
        @constraint(m, (sum(x[id] for id in Ipos) - sum(x[id] for id in Ineg))
                        >=(((lenI - nonzeroNum - tau) * (1 + yijk) /2)
                        - (1 - yijk) * lenI/2 ) )
        @constraint(m, (sum(x[id] for id in Ipos) - sum(x[id] for id in Ineg))
                    <=(((1 + yijk) * lenI/2)
                    - (lenI - nonzeroNum + kappa)*(1 - yijk)/2) )
    end
    return tau, kappa, oneIndices, negOneIndices
end

function addConv2dBinCons!(m::JuMP.Model, xIn::VarOrAff, xOut::VarOrAff,
                        tauList::Array{Float64, 3},
                        kappaList::Array{Float64, 3},
                        oneIndicesList::Array{Array{CartesianIndex{3}, 1}, 3},
                        negOneIndicesList::Array{Array{CartesianIndex{3},1},3},
                        weights::Array{T, 4}, strides::Tuple{Int64, Int64},
                        cb_data) where {T <: Real}
    (x1Len, x2Len, x3Len) = size(xIn)
    (y1Len, y2Len, y3Len) = size(xOut)
    (k1Len, k2Len, channels, filters) = size(weights)
    (s1Len, s2Len) = strides
    xVal = zeros((x1Len, x2Len, x3Len))
    for idx in eachindex(xIn)
        xVal[idx] = aff_callback_value(cb_data, xIn[idx])
    end
    contFlag = true
    yVal = zeros((y1Len, y2Len, y3Len))
    for i in 1:y1Len
        for j in 1:y2Len
            for k in 1:y3Len
                idy = CartesianIndex{3}(i,j,k)
                yVal[idy] = aff_callback_value(cb_data, xOut[idy])
                oneIndices = oneIndicesList[idy]
                negOneIndices = negOneIndicesList[idy]
                nonzeroNum = length(oneIndices) + length(negOneIndices)
                if (nonzeroNum > CUTOFF_CONV2D_BIN)
                    continue
                end
                tau, kappa = tauList[idy], kappaList[idy]
                if (nonzeroNum == 0 || tau >= nonzeroNum || kappa <= -nonzeroNum)
                    continue
                end
                x1Start, x1End = 1 + (i-1)*s1Len, (i-1)*s1Len + k1Len
                x2Start, x2End = 1 + (j-1)*s2Len, (j-1)*s2Len + k2Len
                xValN = xVal[x1Start:min(x1Len, x1End),
                        x2Start:min(x2Len, x2End), :]
                xInN = xIn[x1Start:min(x1Len, x1End),
                        x2Start:min(x2Len, x2End), :]
                con1Val, con2Val = decideViolationConsBin(xValN, yVal[idy],
                                oneIndices, negOneIndices, tau, kappa)
                if (con1Val > 10^(-6))
                    I1pos, I1neg = getFirstBinCutIndices(xValN, yVal[idy],
                                    oneIndices,negOneIndices)
                    lenI1 = length(I1pos) + length(I1neg)
                    con1 = getFirstBinCon(xInN,xOut[idy],I1pos,
                                    I1neg,lenI1,nonzeroNum,tau)
                    MOI.submit(m, MOI.UserCut(cb_data), con1)
                    m.ext[:CUTS].count += 1
                end
                if (con2Val > 10^(-6))
                    I2pos, I2neg = getSecondBinCutIndices(xValN, yVal[idy],
                                    oneIndices,negOneIndices)
                    lenI2 = length(I2pos) + length(I2neg)
                    con2 = getSecondBinCon(xInN,xOut[idy],I2pos,I2neg,
                                    lenI2,nonzeroNum,kappa)
                    MOI.submit(m, MOI.UserCut(cb_data), con2)
                    m.ext[:CUTS].count += 1
                end
            end
        end
    end
    return contFlag
end

function decideViolationConsBin(xVal::Array{R1, 3}, yVal::R2,
                        oneIndices::Array{CartesianIndex{3},1},
                        negOneIndices::Array{CartesianIndex{3},1},
                        tau::R3, kappa::R4) where {R1<:Real, R2<:Real,
                        R3<:Real, R4<:Real}
    nonzeroNum = length(oneIndices) + length(negOneIndices)
    # Initially, I = []
    con1Val = (nonzeroNum + tau) * (1+yVal) / 2
    con2Val = (kappa - nonzeroNum) * (1-yVal) / 2
    for idx in oneIndices
        delta = xVal[idx] - yVal
        if (delta < 0)
            con1Val += delta
        elseif (delta > 0)
            con2Val += delta
        end
    end
    for idx in negOneIndices
        delta = -xVal[idx] - yVal
        if (delta < 0)
            con1Val += delta
        elseif (delta > 0)
            con2Val += delta
        end
    end
    return -con1Val, con2Val
end

# Output positive and negative I^1 for the first constraint in Proposition 3.
function getFirstBinCutIndices(xVal::Array{T, 3}, yVal::U,
                        oneIndices::Array{CartesianIndex{3}, 1},
                        negOneIndices::Array{CartesianIndex{3}, 1}
                        ) where {T<:Real, U<:Real}
    I1pos = Array{CartesianIndex{3}, 1}([])
    I1neg = Array{CartesianIndex{3}, 1}([])
    for idx in oneIndices
        if (xVal[idx] < yVal)
            I1pos = vcat(I1pos, idx)
        end
    end
    for idx in negOneIndices
        if (-xVal[idx] < yVal)
            I1neg = vcat(I1neg, idx)
        end
    end

    return I1pos, I1neg
end

# Output positive and negative I^2 for the second constraint in Proposition 3.
function getSecondBinCutIndices(xVal::Array{T, 3}, yVal::U,
                        oneIndices::Array{CartesianIndex{3}, 1},
                        negOneIndices::Array{CartesianIndex{3}, 1}
                        ) where {T<:Real, U<:Real}
    I2pos = Array{CartesianIndex{3}, 1}([])
    I2neg = Array{CartesianIndex{3}, 1}([])
    for idx in oneIndices
        if (xVal[idx] > yVal)
            I2pos = vcat(I2pos, idx)
        end
    end
    for idx in negOneIndices
        if (-xVal[idx] > yVal)
            I2neg = vcat(I2neg, idx)
        end
    end

    return I2pos, I2neg
end

# Return first constraint with given I, I^+, I^-, tau, kappa as shown
# in Proposition 3. An efficient implementation for user cuts.
function getFirstBinCon(x::VarOrAff, yijk::VarOrAff,
                            Ipos::Array{CartesianIndex{3}, 1},
                            Ineg::Array{CartesianIndex{3}, 1},
                            lenI::Int64,
                            nonzeroNum::Int64,
                            tau::T) where {T <: Real}
    return @build_constraint((sum(x[idx] for idx in Ipos)
                    - sum(x[idx] for idx in Ineg))
                    >=(((lenI - nonzeroNum - tau) * (1 + yijk) /2)
                    - (1 - yijk) * lenI/2 ))
end

# Return second constraint with given I, I^+, I^-, tau, kappa as shown
# in Proposition 3. An efficient implementation for user cuts.
function getSecondBinCon(x::VarOrAff, yijk::VarOrAff,
                            Ipos::Array{CartesianIndex{3}, 1},
                            Ineg::Array{CartesianIndex{3}, 1},
                            lenI::Int64,
                            nonzeroNum::Int64,
                            kappa::T) where {T <: Real}
    return @build_constraint((sum(x[idx] for idx in Ipos)
                - sum(x[idx] for idx in Ineg))
                <=(((1 + yijk) * lenI/2)
                - (lenI - nonzeroNum + kappa)*(1 - yijk)/2))
end
