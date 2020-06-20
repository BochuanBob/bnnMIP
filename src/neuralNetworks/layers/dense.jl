include("layerSetup.jl")
include("activation.jl")
include("denseSetup.jl")
export dense, addDenseCons!

const CUTOFF_DENSE = 10
const CUTOFF_DENSE_PRECUT = 5
const NONZERO_MAX_DENSE = 200
const EXTEND_CUTOFF_DENSE = 10
# The MIP formulation for general dense layer
function dense(m::JuMP.Model, x::VarOrAff,
               weights::Array{T, 2}, bias::Array{U, 1},
               upper::Array{V, 1}, lower::Array{W, 1};
               actFunc="", image=true, preCut=true, extend=true
               ) where{T<:Real, U<:Real, V<:Real, W<:Real}
    initNN!(m)
    count = m.ext[:NN].count
    m.ext[:NN].count += 1
    (yLen, xLen) = size(weights)
    if (length(bias) != yLen)
        error("The sizes of weights and bias don't match!")
    end

    @constraint(m, x .<= upper)
    @constraint(m, x .>= lower)

    tauList = zeros(yLen)
    kappaList = zeros(yLen)
    nonzeroIndicesList = Array{Array{Int64, 1}, 1}(undef, yLen)
    uNewList = Array{Array{Float64, 1}, 1}(undef, yLen)
    lNewList = Array{Array{Float64, 1}, 1}(undef, yLen)
    if (actFunc=="Sign")
        z = @variable(m, [1:yLen], binary=true,
                    base_name="z_$count")
        y = @variable(m, [1:yLen],
                    base_name="y_$count")
        @constraint(m, 2 .* z .- 1 .== y)
        # y = @expression(m, 2 .* z .- 1)
        for i in 1:yLen
            tauList[i], kappaList[i], nonzeroIndicesList[i],
                uNewList[i], lNewList[i] = neuronDenseSign(m, x, y[i],
                weights[i, :], bias[i],
                upper, lower, image=image, preCut=preCut, extend=extend)
        end
    elseif (actFunc == "")
        y = @variable(m, [1:yLen],
                    base_name="y_$count")
        @constraint(m, [i=1:yLen], y[i] ==
                    bias[i] + sum(weights[i,j] * x[j] for j in 1:xLen))
    else
        error("Not supported activation functions for dense layer.")
    end
    return y, tauList, kappaList, nonzeroIndicesList, uNewList, lNewList
end

# A MIP formulation for a single neuron.
function neuronDenseSign(m::JuMP.Model, x::VarOrAff, yi::VarOrAff,
                wVec::Array{T, 1}, b::U,
                upper::Array{V, 1}, lower::Array{W, 1}; image=true, preCut=true
                , extend=true) where{T<:Real, U<:Real, V<:Real, W<:Real}
    # initNN!(m)
    posIndices = findall(wVec .> 0)
    negIndices = findall(wVec .< 0)
    nonzeroIndices = union(posIndices, negIndices)
    nonzeroNum = length(nonzeroIndices)
    if (nonzeroNum == 0)
        if (b >= 0)
            @constraint(m, yi == 1)
        else
            @constraint(m, yi == -1)
        end
        return b, b, nonzeroIndices, upper, lower
    end
    if (image)
        wSum = sum(wVec)
        tau, kappa = getTauAndKappa(nonzeroNum, 510*b + wSum)
        tau = (tau - wSum) / 510
        kappa = (kappa - wSum) / 510
    else
        tau, kappa= b, b
    end

    if (extend && nonzeroNum <= EXTEND_CUTOFF_DENSE)
        x1 = @variable(m, [1:nonzeroNum])
        nonzeroList = nonzeroIndices
        @constraint(m, x1 .>= lower .* (-yi+1) / 2)
        @constraint(m, x1 .<= upper .* (-yi+1) / 2)
        @constraint(m, [i=1:nonzeroNum], x[nonzeroList[i]]-x1[i] >=
                        lower .* (yi+1) / 2)
        @constraint(m, [i=1:nonzeroNum], x[nonzeroList[i]]-x1[i] <=
                        upper .* (yi+1) / 2)
        @constraint(m, sum(weightVec[nonzeroList[i]] * x1[i]
                        for i in 1:nonzeroNum) + (kappa/2) * (-yi+1) <= 0)
        @constraint(m, sum(weightVec[nonzeroList[i]] * (x[nonzeroList[i]]-x1[i])
                        for i in 1:nonzeroNum) + (tau/2) * (yi+1) >= 0)
        return tau, kappa, nonzeroIndices, upper, lower
    end
    uNew, lNew = transformProc(negIndices, upper, lower)
    if (preCut && (nonzeroNum <= CUTOFF_DENSE_PRECUT) )
        IsetAll = collect(powerset(nonzeroIndices))
        IsetAll = IsetAll[2:length(IsetAll)]
    else
        IsetAll = [union(posIndices, negIndices)]
    end
    for Iset in IsetAll
        IsetC = setdiff(nonzeroIndices, Iset)
        w = wVec
        @constraint(m, 2 * (sum(w[i]*x[i] for i in Iset) +
                            sum(w[i] * uNew[i] for i in IsetC) + tau) >=
                            ((sum(w[i]*lNew[i] for i in Iset) +
                            sum(w[i] * uNew[i] for i in IsetC) + tau) * (1 - yi)) )
        @constraint(m, 2 * sum(w[i]*(uNew[i] - x[i]) for i in Iset) >=
                            ((sum(w[i]*uNew[i] for i in Iset) +
                            sum(w[i]*lNew[i] for i in IsetC) + kappa) * (1 - yi)) )
    end
    return tau, kappa, nonzeroIndices, uNew, lNew
end


function addDenseCons!(m::JuMP.Model, xIn::VarOrAff, xVal::Array{Float64, 1},
                        xOut::VarOrAff,
                        weights::Array{Float64, 2},tauList::Array{Float64, 1},
                        kappaList::Array{Float64, 1},
                        nonzeroIndicesList::Array{Array{Int64, 1}},
                        uNewList::Array{Array{Float64, 1}, 1},
                        lNewList::Array{Array{Float64, 1}, 1},
                        cb_data; image=true)
    yLen, xLen = length(xOut), length(xIn)
    # xVal = zeros(xLen)
    # for j in 1:xLen
    #     xVal[j] = Float64(JuMP.callback_value(cb_data, xIn[j]))
    # end
    contFlag = true

    yVal = zeros(yLen)
    K = 2
    iter = 0
    for i in 1:yLen
        # if (iter > K)
        #     break
        # end
        yVal[i] = aff_callback_value(cb_data, xOut[i])
        if (-0.99 >= yVal[i] || yVal[i] >= 0.99)
            continue
        end
        wVec = weights[i, :]
        nonzeroIndices = nonzeroIndicesList[i]
        nonzeroNum = length(nonzeroIndices)
        tau, kappa = tauList[i], kappaList[i]
        uNew, lNew = uNewList[i], lNewList[i]
        end
        m.ext[:BENCH_CONV2D].time += time
        if (nonzeroNum == 0 || nonzeroNum > NONZERO_MAX_DENSE)
            continue
        end
        con1Val, con2Val = decideViolationCons(xVal, yVal[i],nonzeroIndices,
                            wVec, tau, kappa, uNew,lNew)

        m.ext[:TEST_CONSTRAINTS].count += 2
        if (con1Val > 0.01)
            I1 = getFirstCutIndices(xVal, yVal[i],nonzeroIndices,wVec,
                                    uNew,lNew)
            if (length(I1) <= CUTOFF_DENSE)
                con1 = getFirstCon(xIn, xOut[i], I1,
                            nonzeroIndices, wVec, tau, uNew, lNew)

                # assertFirstCon(xVal, yVal[i], I1,
                #             nonzeroIndices, wVec, tau, uNew, lNew)
                MOI.submit(m, MOI.UserCut(cb_data), con1)
                m.ext[:CUTS].count += 1
                iter += 1
            end

        end
        if (con2Val > 0.01)
            I2 = getSecondCutIndices(xVal, yVal[i],nonzeroIndices,wVec,
                                    uNew,lNew)
            if (length(I2) <= CUTOFF_DENSE)
                con2 = getSecondCon(xIn, xOut[i], I2,
                            nonzeroIndices, wVec, kappa, uNew, lNew)
                # assertSecondCon(xVal, yVal[i], I2,
                #             nonzeroIndices, wVec, kappa, uNew, lNew)
                MOI.submit(m, MOI.UserCut(cb_data), con2)
                m.ext[:CUTS].count += 1
                iter += 1
            end
        end

    end

    return yVal, contFlag
end


# A transformation for Fourier-Motzkin procedure.
function transformProc(negIndices::Array{Int64, 1},
            upper::Array{T, 1}, lower::Array{U, 1}) where {T<:Real, U<:Real}
    upperNew = deepcopy(upper)
    lowerNew = deepcopy(lower)
    for i in negIndices
        upperNew[i] = lower[i]
        lowerNew[i] = upper[i]
    end
    return Float64.(upperNew), Float64.(lowerNew)
end

function decideViolationCons(xVal::Array{Float64, 1}, yVal::Float64,
                        nonzeroIndices::Array{Int64, 1},
                        w::Array{Float64, 1}, tau::Float64, kappa::Float64,
                        upper::Array{Float64, 1}, lower::Array{Float64, 1}
                        )
    # Initially, I = []
    con1Val = 2 * (sum(w[i] * upper[i] for i in nonzeroIndices) + tau) -
                (sum(w[i] * upper[i] for i in nonzeroIndices) + tau) * (1 - yVal)
    con2Val = - (sum(w[i] * lower[i] for i in nonzeroIndices) + kappa) * (1 - yVal)
    for i in nonzeroIndices
        con1Delta = 2*w[i]*(xVal[i] - upper[i]) - w[i]*(lower[i]-upper[i])*(1-yVal)
        con2Delta = 2*w[i]*(upper[i]-xVal[i]) - w[i]*(upper[i]-lower[i])*(1-yVal)
        con1Val = con1Val + min(0, con1Delta)
        con2Val = con2Val + min(0, con2Delta)
    end
    return -con1Val, -con2Val
end

# Output I^1 for the first constraint in Proposition 1
function getFirstCutIndices(xVal::Array{Float64, 1}, yVal::Float64,
                        nonzeroIndices::Array{Int64, 1},
                        w::Array{Float64, 1},
                        upper::Array{Float64, 1}, lower::Array{Float64, 1}
                        )
    I1 = Array{Int64, 1}([])
    for i in nonzeroIndices
        if (2*w[i]*(xVal[i] - upper[i]) < w[i]*(lower[i]-upper[i])*(1-yVal))
            append!(I1, i)
        end
    end
    return I1
end

# Output I^2 for the second constraint in Proposition 1
function getSecondCutIndices(xVal::Array{Float64, 1}, yVal::Float64,
                        nonzeroIndices::Array{Int64, 1},
                        w::Array{Float64, 1},
                        upper::Array{Float64, 1}, lower::Array{Float64, 1}
                        )
    I2 = Array{Int64, 1}([])
    for i in nonzeroIndices
        if (2*w[i]*(upper[i]-xVal[i]) < w[i]*(upper[i]-lower[i])*(1-yVal))
            append!(I2, i)
        end
    end
    return I2
end

# Return first constraint with given I, I^+, I^-, tau, kappa as shown
# in Proposition 1. An efficient implementation for user cuts.
function getFirstCon(x::VarOrAff, yi::VarOrAff,
                            Iset::Array{Int64, 1},
                            nonzeroIndices::Array{Int64, 1},
                            w::Array{Float64, 1},b::Float64,
                            upper::Array{Float64, 1}, lower::Array{Float64, 1}
                            )
    IsetC = setdiff(nonzeroIndices, Iset)
    return @build_constraint(2 * (sum(w[i]*x[i] for i in Iset) +
                        sum(w[i] * upper[i] for i in IsetC) + b) >=
                        (sum(w[i]*lower[i] for i in Iset) +
                        sum(w[i] * upper[i] for i in IsetC) + b) * (1 - yi)
                        )
end

# Return second constraint with given I, I^+, I^-, tau, kappa as shown
# in Proposition 1. An efficient implementation for user cuts.
function getSecondCon(x::VarOrAff, yi::VarOrAff,
                            Iset::Array{Int64, 1},
                            nonzeroIndices::Array{Int64, 1},
                            w::Array{Float64, 1},b::Float64,
                            upper::Array{Float64, 1}, lower::Array{Float64, 1}
                            )
    IsetC = setdiff(nonzeroIndices, Iset)
    return @build_constraint(2 * sum(w[i]*(upper[i] - x[i]) for i in Iset) >=
                        (sum(w[i]*upper[i] for i in Iset) +
                        sum(w[i]*lower[i] for i in IsetC) + b) * (1 - yi)
                        )
end


function assertFirstCon(x, yi,
                            Iset::Array{Int64, 1},
                            nonzeroIndices::Array{Int64, 1},
                            w::Array{Float64, 1},b::Float64,
                            upper::Array{Float64, 1}, lower::Array{Float64, 1}
                            )
    # println(2 * (sum(w[Iset] .* x[Iset]) +
    #                     sum(w[IsetC] .* upper[IsetC]) + b) -
    #                     ((sum(w[Iset] .* lower[Iset]) +
    #                     sum(w[IsetC] .* upper[IsetC]) + b) * (1 - yi) ))
    IsetC = setdiff(nonzeroIndices, Iset)
    @assert(2 * (sum(w[Iset] .* x[Iset]) +
                        sum(w[IsetC] .* upper[IsetC]) + b) <
                        (sum(w[Iset] .* lower[Iset]) +
                        sum(w[IsetC] .* upper[IsetC]) + b) * (1 - yi)
                        )
    return
end

function assertSecondCon(x, yi,
                            Iset::Array{Int64, 1},
                            nonzeroIndices::Array{Int64, 1},
                            w::Array{Float64, 1},b::Float64,
                            upper::Array{Float64, 1}, lower::Array{Float64, 1}
                            )
    IsetC = setdiff(nonzeroIndices, Iset)
    # println(2 * sum(w[Iset] .* (upper[Iset] - x[Iset])) -
    #                     (sum(w[Iset] .* upper[Iset]) +
    #                     sum(w[IsetC] .* lower[IsetC]) + b) * (1 - yi) )
    @assert(2 * sum(w[Iset] .* (upper[Iset] - x[Iset])) <
                        (sum(w[Iset] .* upper[Iset]) +
                        sum(w[IsetC] .* lower[IsetC]) + b) * (1 - yi)
                        )
    return
end
