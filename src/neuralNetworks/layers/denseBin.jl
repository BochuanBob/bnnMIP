export denseBin, addDenseBinCons!

const CUTOFF_DENSE_BIN = 100
const CUTOFF_DENSE_BIN_PRECUT = 10
const NONZERO_MAX_DENSE_BIN = 1000
const EXTEND_CUTOFF_DENSE_BIN = 10
# A fully connected layer with sign() or
# without activation function.
# Each entry of weights must be -1, 0, 1.
function denseBin(m::JuMP.Model, x::VarOrAff,
               weights::Array{Float64, 2}, bias::Array{Float64, 1},
               upper::Array{Float64, 1}, lower::Array{Float64, 1};
               takeSign=false, image=true, preCut=true,
               cuts=true, extend=false, layer=0, forward=true)
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
        z = @variable(m, [1:yLen], binary=true,
                    base_name="z_$count")
        # if (cuts)
        #     MOI.set.(Ref(m), Ref(Gurobi.VariableAttribute("BranchPriority")),
        #             z, Ref(layer))
        # end
        # y = @expression(m, 2 .* z .- 1)
        y = @variable(m, [1:yLen],
                    base_name="y_$count")
        @constraint(m, 2 .* z .- 1 .== y)
        for i in 1:yLen
            tauList[i], kappaList[i], oneIndicesList[i], negOneIndicesList[i] =
                neuronSign(m, x, y[i], weights[i, :], bias[i], upper, lower,
                        image=image, preCut=preCut, cuts=cuts, extend=extend,
                        forward=forward)
        end
    else
        y = @variable(m, [1:yLen],
                    base_name="y_$count")
        z = y
        @constraint(m, [i=1:yLen], y[i] ==
                    bias[i] + sum(weights[i,j] * x[j] for j in 1:xLen))
    end
    return y, z, tauList, kappaList, oneIndicesList, negOneIndicesList
end

# A MIP formulation for a single neuron.
function neuronSign(m::JuMP.Model, x::VarOrAff, yi::VarOrAff,
                weightVec::Array{T, 1}, b::U,
                upper::Array{Float64, 1}, lower::Array{Float64, 1};
                image=true, preCut=true, cuts=true,
                extend=true, forward=true) where{T<:Real, U<:Real}
    # initNN!(m)

    if (forward)
        oneIndices = findall((weightVec .== 1) .& (upper .!= lower))
        negOneIndices = findall((weightVec .== -1) .& (upper .!= lower))
        fixIndices = findall(upper .== lower)
        for i in fixIndices
            b += weightVec[i] * lower[i]
        end
    else
        oneIndices = findall(weightVec .== 1)
        negOneIndices = findall(weightVec .== -1)
    end
    nonzeroNum = length(oneIndices) + length(negOneIndices)

    if (nonzeroNum == 0)
        if (b >= 0)
            @constraint(m, yi == 1)
        else
            @constraint(m, yi == -1)
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
        @constraint(m, yi == 1)
        return tau, kappa, oneIndices, negOneIndices
    end
    if (kappa <= -nonzeroNum)
        @constraint(m, yi == -1)
        return tau, kappa, oneIndices, negOneIndices
    end

    # if ((preCut || cuts) && (nonzeroNum <= CUTOFF_DENSE_BIN_PRECUT))
    #     (var, _) = collect(yi.terms)[1]
    #     MOI.set(m, Gurobi.VariableAttribute("BranchPriority"), var, -1)
    # end

    if (extend && nonzeroNum <= EXTEND_CUTOFF_DENSE_BIN)
        x1 = @variable(m, [1:nonzeroNum])
        nonzeroList = union(oneIndices, negOneIndices)
        @constraint(m, x1 .>= -(-yi+1) / 2)
        @constraint(m, x1 .<= (-yi+1) / 2)
        @constraint(m, [i=1:nonzeroNum], x[nonzeroList[i]]-x1[i] >= -(yi+1) / 2)
        @constraint(m, [i=1:nonzeroNum], x[nonzeroList[i]]-x1[i] <= (yi+1) / 2)
        @constraint(m, sum(weightVec[nonzeroList[i]] * x1[i]
                        for i in 1:nonzeroNum) + (kappa/2) * (-yi+1) <= 0)
        @constraint(m, sum(weightVec[nonzeroList[i]] * (x[nonzeroList[i]]-x1[i])
                        for i in 1:nonzeroNum) + (tau/2) * (yi+1) >= 0)
        return tau, kappa, oneIndices, negOneIndices
    end

    if (preCut && (nonzeroNum <= CUTOFF_DENSE_BIN_PRECUT) )
        IsetAll = collect(powerset(union(oneIndices, negOneIndices)))
        IsetAll = IsetAll[2:length(IsetAll)]
    else
        IsetAll = [union(oneIndices, negOneIndices)]
    end
    for Iset1 in IsetAll
        Ipos = intersect(Iset1, oneIndices)
        Ineg = intersect(Iset1, negOneIndices)
        lenI = length(Iset1)
        @constraint(m, (sum(x[i] for i in Ipos) - sum(x[i] for i in Ineg))
                        >=(((lenI - nonzeroNum - tau) * (1 + yi) /2) - (1 - yi) * lenI/2 ) )
        @constraint(m, (sum(x[i] for i in Ipos) - sum(x[i] for i in Ineg))
                    <=(((1 + yi) * lenI/2) - (lenI - nonzeroNum + kappa)*(1 - yi)/2) )
    end
    return tau, kappa, oneIndices, negOneIndices
end

function addDenseBinCons!(m::JuMP.Model, opt::Gurobi.Optimizer,
                        xIn::Array{VariableRef, 1}, xVal::Array{Float64, 1},
                        xOut::Array{VariableRef, 1},
                        tauList::Array{Float64, 1},
                        kappaList::Array{Float64, 1},
                        oneIndicesList::Array{Array{Int64, 1}, 1},
                        negOneIndicesList::Array{Array{Int64, 1}, 1},
                        cb_data::Gurobi.CallbackData)
    yLen, xLen = length(xOut), length(xIn)
    contFlag = true
    yVal = zeros(yLen)
    for i in 1:length(xOut)
        yVal[i] = my_callback_value(opt, cb_data, xOut[i])
    end
    # return yVal, contFlag
    for i in 1:yLen
        if (-1 + 10^(-8) >= yVal[i] || yVal[i] >= 1 - 10^(-8))
            continue
        end
        oneIndices = oneIndicesList[i]
        negOneIndices = negOneIndicesList[i]
        nonzeroNum = length(oneIndices) + length(negOneIndices)
        if (nonzeroNum == 0 || nonzeroNum > NONZERO_MAX_DENSE_BIN)
            continue
        end
        tau, kappa = tauList[i], kappaList[i]
        if (tau >= nonzeroNum || kappa <= -nonzeroNum)
            continue
        end

        con1Val, con2Val, count1, count2 =
                    decideViolationConsBin(xVal, yVal[i], oneIndices,
                        negOneIndices, tau, kappa)

        m.ext[:TEST_CONSTRAINTS].count += 2
        # if (2 * count1 > nonzeroNum + tau + 1)
        if (con1Val > 10^(-5))
            I1pos, I1neg = getFirstBinCutIndices(xVal, yVal[i],
                            oneIndices,negOneIndices, tau)
            lenI1 = length(I1pos) + length(I1neg)
            con1 = getFirstBinCon(xIn,xOut[i],I1pos,I1neg,lenI1,nonzeroNum,tau)
            # assertFirstBinCon(xVal,yVal[i],I1pos,I1neg,lenI1,nonzeroNum,tau)
            MOI.submit(m, MOI.UserCut(cb_data), con1)
            m.ext[:CUTS].count += 1
        end
        # if (2 * count2 > nonzeroNum - kappa + 1)
        if (con2Val > 10^(-5))
            I2pos, I2neg = getSecondBinCutIndices(xVal, yVal[i],
                            oneIndices,negOneIndices, kappa)
            lenI2 = length(I2pos) + length(I2neg)
            con2 = getSecondBinCon(xIn,xOut[i],I2pos,I2neg,lenI2,nonzeroNum,kappa)
            # assertSecondBinCon(xVal,yVal[i],I2pos,I2neg,lenI2,nonzeroNum,kappa)
            MOI.submit(m, MOI.UserCut(cb_data), con2)
            m.ext[:CUTS].count += 1
        end
    end
    return yVal, contFlag
end

function decideViolationConsBin(xVal::Array{Float64, 1}, yVal::Float64,
                        oneIndices::Array{Int64, 1},
                        negOneIndices::Array{Int64, 1},
                        tau::Float64, kappa::Float64)
    nonzeroNum = length(oneIndices) + length(negOneIndices)
    count1 = 0
    count2 = 0
    # Initially, I = []
    con1Val = (nonzeroNum + tau) * (1+yVal) / 2
    con2Val = (kappa - nonzeroNum) * (1-yVal) / 2
    for i in oneIndices
        if (xVal[i] < 1 - 10^(-8) && xVal[i] > -1 + 10^(-8))
            continue
        end
        delta = xVal[i] - yVal
        if (delta < 0)
            con1Val += delta
            count1 += 1
        elseif (delta > 0)
            con2Val += delta
            count2 += 1
        end
    end
    for i in negOneIndices
        if (xVal[i] < 1 - 10^(-8) && xVal[i] > -1 + 10^(-8))
            continue
        end
        delta = -xVal[i] - yVal
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
function getFirstBinCutIndices(xVal::Array{Float64, 1}, yVal::Float64,
                        oneIndices::Array{Int64, 1},
                        negOneIndices::Array{Int64, 1}, tau::Float64)
    I1pos = Array{Int64, 1}(undef, length(oneIndices))
    I1neg = Array{Int64, 1}(undef, length(negOneIndices))
    count1 = 0
    count2 = 0
    for i in oneIndices
        if (xVal[i] < 1 - 10^(-8) && xVal[i] > -1 + 10^(-8))
            continue
        end
        if (xVal[i] < yVal)
            count1 += 1
            I1pos[count1] = i
        end
    end
    for i in negOneIndices
        if (xVal[i] < 1 - 10^(-8) && xVal[i] > -1 + 10^(-8))
            continue
        end
        if (-xVal[i] < yVal)
            count2 += 1
            I1neg[count2] = i
        end
    end

    # if (2*(count1 + count2) > nonzeroNum + tau + 1)
    #     change = 2*(count1 + count2) - (nonzeroNum + tau + 1)
    #     change = Int64(floor(0.5 * change))
    #     if (count1 < change)
    #         count1 = 0
    #         count2 -= (change - count1)
    #     else
    #         count1 -= change
    #     end
    # end

    I1pos = I1pos[1:count1]
    I1neg = I1neg[1:count2]
    return I1pos, I1neg
end

# Output positive and negative I^2 for the second constraint in Proposition 3.
function getSecondBinCutIndices(xVal::Array{Float64, 1}, yVal::Float64,
                        oneIndices::Array{Int64, 1},
                        negOneIndices::Array{Int64, 1}, kappa::Float64)
    I2pos = Array{Int64, 1}(undef, length(oneIndices))
    I2neg = Array{Int64, 1}(undef, length(negOneIndices))
    count1 = 0
    count2 = 0
    for i in oneIndices
        if (xVal[i] < 1 - 10^(-8) && xVal[i] > -1 + 10^(-8))
            continue
        end
        if (xVal[i] > yVal)
            count1 += 1
            I2pos[count1] = i
        end
    end
    for i in negOneIndices
        if (xVal[i] < 1 - 10^(-8) && xVal[i] > -1 + 10^(-8))
            continue
        end
        if (-xVal[i] > yVal)
            count2 += 1
            I2neg[count2] = i
        end
    end
    # if (2*(count1 + count2) > nonzeroNum - kappa + 1)
    #     change = 2*(count1 + count2) - (nonzeroNum - kappa + 1)
    #     change = Int64(floor(0.5 * change))
    #     if (count1 < change)
    #         count1 = 0
    #         count2 -= (change - count1)
    #     else
    #         count1 -= change
    #     end
    # end
    I2pos = I2pos[1:count1]
    I2neg = I2neg[1:count2]
    return I2pos, I2neg
end

# Return first constraint with given I, I^+, I^-, tau, kappa as shown
# in Proposition 3. An efficient implementation for user cuts.
function getFirstBinCon(x::VarOrAff, yi::VarOrAff,
                            Ipos::Array{Int64, 1},
                            Ineg::Array{Int64, 1},
                            lenI::Int64,
                            nonzeroNum::Int64,
                            tau::Float64)
    return @build_constraint((sum(x[i] for i in Ipos) - sum(x[i] for i in Ineg))
                    >=(((lenI - nonzeroNum - tau) * (1 + yi) /2) - (1 - yi) * lenI/2 ))
end

# Return second constraint with given I, I^+, I^-, tau, kappa as shown
# in Proposition 3. An efficient implementation for user cuts.
function getSecondBinCon(x::VarOrAff, yi::VarOrAff,
                            Ipos::Array{Int64, 1},
                            Ineg::Array{Int64, 1},
                            lenI::Int64,
                            nonzeroNum::Int64,
                            kappa::Float64)
    return @build_constraint((sum(x[i] for i in Ipos) - sum(x[i] for i in Ineg))
                <=(((1 + yi) * lenI/2) - (lenI - nonzeroNum + kappa)*(1 - yi)/2))
end


function assertFirstBinCon(x, yi,
                            Ipos::Array{Int64, 1},
                            Ineg::Array{Int64, 1},
                            lenI::Int64,
                            nonzeroNum::Int64,
                            tau::Float64)
    @assert((sum(x[Ipos]) - sum(x[Ineg]))
                    < (((lenI - nonzeroNum - tau) * (1 + yi) /2) - (1 - yi) * lenI/2 ))
    return
end

function assertSecondBinCon(x, yi,
                            Ipos::Array{Int64, 1},
                            Ineg::Array{Int64, 1},
                            lenI::Int64,
                            nonzeroNum::Int64,
                            kappa::Float64)
    @assert((sum(x[Ipos]) - sum(x[Ineg]))
                > (((1 + yi) * lenI/2) - (lenI - nonzeroNum + kappa)*(1 - yi)/2))
    return
end
