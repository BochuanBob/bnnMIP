export conv2dBinSign

# A conv2d layer with sign().
# Each entry of weights must be -1, 0, 1.
function conv2dBinSign(m::JuMP.Model, x::VarOrAff,
                weights::Array{T, 4}, bias::Array{U, 1},
                strides::Tuple{Int64, Int64}; padding="valid",
                image=true, preCut=true, cuts=true, layer=0
                ) where{T<:Real, U<:Real}
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
    tauList = zeros(y3Len)
    kappaList = zeros(y3Len)
    oneIndicesList = Array{Array{CartesianIndex{3}, 1}, 3}(undef, (k1Len, k2Len, y3Len))
    negOneIndicesList = Array{Array{CartesianIndex{3}, 1}, 3}(undef, (k1Len, k2Len, y3Len))
    for i in 1:k1Len
        for j in 1:k2Len
            for k in 1:y3Len
                oneIndicesList[i,j,k] = findall(weights[1:i, 1:j, :, k] .== 1)
                negOneIndicesList[i,j,k] = findall(weights[1:i, 1:j, :, k] .== -1)
            end
        end
    end
    z = @variable(m, [1:y1Len, 1:y2Len, 1:y3Len], binary=true,
                base_name="z_$count")
    # y = @expression(m, 2 .* z .- 1)
    # if (cuts)
    #     MOI.set.(Ref(m), Ref(Gurobi.VariableAttribute("BranchPriority")),
    #             z, Ref(layer))
    # end
    y = @variable(m, [1:y1Len, 1:y2Len, 1:y3Len],
                base_name="y_$count")
    @constraint(m, 2 .* z .- 1 .== y)
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
                tauList[k], kappaList[k] = neuronSign(m, xN, y[i,j,k],
                            weight, bias[k], image=image, preCut=preCut,
                            cuts=cuts)
            end
        end
    end
    return y, z, tauList, kappaList, oneIndicesList, negOneIndicesList
end

# A MIP formulation for a single neuron.
function neuronSign(m::JuMP.Model, x::VarOrAff, yijk::VarOrAff,
                weightVec::Array{T, 3}, b::U;
                image=true, preCut=true, cuts=true) where{T<:Real, U<:Real}
    # initNN!(m)
    oneIndices = findall(weightVec .== 1)
    negOneIndices = findall(weightVec .== -1)
    nonzeroNum = length(oneIndices) + length(negOneIndices)

    # For comparison to not handling discontinuity.
    if (image)
        tau, kappa = getTauAndKappa(nonzeroNum, b)
    else
        tau, kappa = b, b
    end

    if (nonzeroNum == 0 || size(x) == (0,0,0))
        if (b >= 0)
            @constraint(m, yijk == 1)
        else
            @constraint(m, yijk == -1)
        end
        return tau, kappa
    end
    if (tau >= nonzeroNum)
        @constraint(m, yijk == 1)
        return tau, kappa
    end
    if (kappa <= -nonzeroNum)
        @constraint(m, yijk == -1)
        return tau, kappa
    end
    # if ((preCut || cuts) && (nonzeroNum <= CUTOFF_CONV2D_BIN_PRECUT))
    #     (var, _) = collect(yijk.terms)[1]
    #     MOI.set(m, Gurobi.VariableAttribute("BranchPriority"), var, -1)
    # end
    if (preCut && (nonzeroNum <= CUTOFF_CONV2D_BIN_PRECUT) )
        IsetAll = collect(powerset(union(oneIndices, negOneIndices)))
        IsetAll = IsetAll[2:length(IsetAll)]
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
    return tau, kappa
end

function addConv2dBinCons!(m::JuMP.Model, opt::Gurobi.Optimizer,
                        xIn::Array{VariableRef, 3},
                        xVal::Array{Float64, 3},
                        xOut::Array{VariableRef, 3},
                        tauList::Array{Float64, 1},
                        kappaList::Array{Float64, 1},
                        oneIndicesList::Array{Array{CartesianIndex{3}, 1}, 3},
                        negOneIndicesList::Array{Array{CartesianIndex{3},1},3},
                        weights::Array{Float64, 4}, strides::Tuple{Int64, Int64},
                        cb_data::Gurobi.CallbackData)
    (x1Len, x2Len, x3Len) = size(xIn)
    (y1Len, y2Len, y3Len) = size(xOut)
    (k1Len, k2Len, channels, filters) = size(weights)
    (s1Len, s2Len) = strides
    contFlag = true
    yVal = zeros((y1Len, y2Len, y3Len))
    for i in 1:y1Len
        for j in 1:y2Len
            for k in 1:y3Len
                yVal[i,j,k] = my_callback_value(opt, cb_data, xOut[i,j,k])
            end
        end
    end
    for i in 1:y1Len
        for j in 1:y2Len
            for k in 1:y3Len
                if (-1 + 10^(-8) >= yVal[i,j,k] || yVal[i,j,k] >= 1 - 10^(-8))
                    continue
                end
                x1Start, x1End = 1 + (i-1)*s1Len, (i-1)*s1Len + k1Len
                x2Start, x2End = 1 + (j-1)*s2Len, (j-1)*s2Len + k2Len
                x1NEnd = min(x1Len, x1End)
                x2NEnd = min(x2Len, x2End)
                oneIndices = oneIndicesList[(x1NEnd-x1Start+1),(x2NEnd-x2Start+1),k]
                negOneIndices = negOneIndicesList[(x1NEnd-x1Start+1),(x2NEnd-x2Start+1),k]
                nonzeroNum = length(oneIndices) + length(negOneIndices)

                if (nonzeroNum == 0 || nonzeroNum > CUTOFF_CONV2D_BIN)
                    continue
                end
                tau, kappa = tauList[k], kappaList[k]
                if (tau >= nonzeroNum || kappa <= -nonzeroNum)
                    continue
                end
                xValN = xVal[x1Start:x1NEnd,
                        x2Start:x2NEnd, :]
                xInN = xIn[x1Start:x1NEnd,
                        x2Start:x2NEnd, :]
                con1Val, con2Val, count1, count2 = decideViolationConsBin(xValN, yVal[i,j,k],
                                oneIndices, negOneIndices, tau, kappa)
                if (2 * count1 > nonzeroNum + tau + 1)
                    I1pos, I1neg = getFirstBinCutIndices(xValN, yVal[i,j,k],
                                    oneIndices,negOneIndices)
                    lenI1 = length(I1pos) + length(I1neg)
                    con1 = getFirstBinCon(xInN,xOut[i,j,k],I1pos,
                                    I1neg,lenI1,nonzeroNum,tau)
                    MOI.submit(m, MOI.UserCut(cb_data), con1)
                    m.ext[:CUTS].count += 1
                end
                if (2 * count2 > nonzeroNum - kappa + 1)
                    I2pos, I2neg = getSecondBinCutIndices(xValN, yVal[i,j,k],
                                    oneIndices,negOneIndices)
                    lenI2 = length(I2pos) + length(I2neg)
                    con2 = getSecondBinCon(xInN,xOut[i,j,k],I2pos,I2neg,
                                    lenI2,nonzeroNum,kappa)
                    MOI.submit(m, MOI.UserCut(cb_data), con2)
                    m.ext[:CUTS].count += 1
                end
            end
        end
    end
    return yVal, contFlag
end

function decideViolationConsBin(xVal::Array{Float64, 3}, yVal::Float64,
                        oneIndices::Array{CartesianIndex{3},1},
                        negOneIndices::Array{CartesianIndex{3},1},
                        tau::Float64, kappa::Float64)
    nonzeroNum = length(oneIndices) + length(negOneIndices)
    # Initially, I = []
    con1Val = (nonzeroNum + tau) * (1+yVal) / 2
    con2Val = (kappa - nonzeroNum) * (1-yVal) / 2
    count1 = 0
    count2 = 0
    for idx in oneIndices
        delta = xVal[idx] - yVal
        if (xVal[idx]<= 1 - 10^(-8) && xVal[idx] >= -1 + 10^(-8))
            continue
        end
        if (delta < 0)
            con1Val += delta
            count1 += 1
        elseif (delta > 0)
            con2Val += delta
            count2 += 1
        end
    end
    for idx in negOneIndices
        delta = -xVal[idx] - yVal
        if (xVal[idx]<= 1 - 10^(-8) && xVal[idx] >= -1 + 10^(-8))
            continue
        end
        if (delta < 0)
            con1Val += delta
            count1 += 1
        elseif (delta > 0)
            con2Val += delta
            count2 += 1
        end
    end
    return -con1Val, con2Val, count1, count2
end

# Output positive and negative I^1 for the first constraint in Proposition 3.
function getFirstBinCutIndices(xVal::Array{Float64, 3}, yVal::Float64,
                        oneIndices::Array{CartesianIndex{3}, 1},
                        negOneIndices::Array{CartesianIndex{3}, 1}
                        )
    oneNum = length(oneIndices)
    negOneNum = length(negOneIndices)
    I1pos = Array{CartesianIndex{3}, 1}(undef, oneNum)
    I1neg = Array{CartesianIndex{3}, 1}(undef, negOneNum)
    count1 = 0
    count2 = 0
    for idx in oneIndices
        if (xVal[idx]<= 1 - 10^(-8) && xVal[idx] >= -1 + 10^(-8))
            continue
        end
        if (xVal[idx] < yVal)
            count1 += 1
            I1pos[count1] = idx
        end
    end
    for idx in negOneIndices
        if (xVal[idx]<= 1 - 10^(-8) && xVal[idx] >= -1 + 10^(-8))
            continue
        end
        if (-xVal[idx] < yVal)
            count2 += 1
            I1neg[count2] = idx
        end
    end
    I1pos = I1pos[1:count1]
    I1neg = I1neg[1:count2]
    return I1pos, I1neg
end

# Output positive and negative I^2 for the second constraint in Proposition 3.
function getSecondBinCutIndices(xVal::Array{Float64, 3}, yVal::Float64,
                        oneIndices::Array{CartesianIndex{3}, 1},
                        negOneIndices::Array{CartesianIndex{3}, 1}
                        )
    oneNum = length(oneIndices)
    negOneNum = length(negOneIndices)
    I2pos = Array{CartesianIndex{3}, 1}(undef, oneNum)
    I2neg = Array{CartesianIndex{3}, 1}(undef, negOneNum)
    count1 = 0
    count2 = 0
    for idx in oneIndices
        if (xVal[idx]<= 1 - 10^(-8) && xVal[idx] >= -1 + 10^(-8))
            continue
        end
        if (xVal[idx] > yVal)
            count1 += 1
            I2pos[count1] = idx
        end
    end
    for idx in negOneIndices
        if (xVal[idx]<= 1 - 10^(-8) && xVal[idx] >= -1 + 10^(-8))
            continue
        end
        if (-xVal[idx] > yVal)
            count2 += 1
            I2neg[count2] = idx
        end
    end
    I2pos = I2pos[1:count1]
    I2neg = I2neg[1:count2]
    return I2pos, I2neg
end

# Return first constraint with given I, I^+, I^-, tau, kappa as shown
# in Proposition 3. An efficient implementation for user cuts.
function getFirstBinCon(x::VarOrAff, yijk::VarOrAff,
                            Ipos::Array{CartesianIndex{3}, 1},
                            Ineg::Array{CartesianIndex{3}, 1},
                            lenI::Int64,
                            nonzeroNum::Int64,
                            tau::Float64)
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
                            kappa::Float64)
    return @build_constraint((sum(x[idx] for idx in Ipos)
                - sum(x[idx] for idx in Ineg))
                <=(((1 + yijk) * lenI/2)
                - (lenI - nonzeroNum + kappa)*(1 - yijk)/2))
end
