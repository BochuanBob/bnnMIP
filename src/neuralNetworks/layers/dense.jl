include("layerSetup.jl")
include("activation.jl")
include("denseSetup.jl")
export dense, addDenseCons!

# The MIP formulation for general dense layer
function dense(m::JuMP.Model, x::VarOrAff,
               weights::Array{T, 2}, bias::Array{U, 1},
               upper::Array{V, 1}, lower::Array{W, 1};
               actFunc="", image=true
               ) where{T<:Real, U<:Real, V<:Real, W<:Real}
    initNN!(m)
    count = m.ext[:NN].count
    m.ext[:NN].count += 1
    (yLen, xLen) = size(weights)
    if (length(bias) != yLen)
        error("The sizes of weights and bias don't match!")
    end
    if (~image)
        @constraint(m, x .<= upper)
        @constraint(m, x .>= lower)
    end
    tauList = zeros(yLen)
    kappaList = zeros(yLen)
    nonzeroIndicesList = Array{Array{Int64, 1}, 1}(undef, yLen)
    uNewList = Array{Array{Float64, 1}, 1}(undef, yLen)
    lNewList = Array{Array{Float64, 1}, 1}(undef, yLen)
    if (actFunc=="Sign")
        z = @variable(m, [1:yLen], binary=true,
                    base_name="z_$count")
        y = @expression(m, 2 .* z .- 1)
        for i in 1:yLen
            tauList[i], kappaList[i], nonzeroIndicesList[i],
                uNewList[i], lNewList[i] = neuronDenseSign(m, x, y[i],
                weights[i, :], bias[i],
                upper, lower, image=image)
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
                upper::Array{V, 1}, lower::Array{W, 1}; image=true
                ) where{T<:Real, U<:Real, V<:Real, W<:Real}
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
    uNew, lNew = transformProc(negIndices, upper, lower)
    Iset = union(posIndices, negIndices)
    IsetC = setdiff(nonzeroIndices, Iset)
    w = wVec
    @constraint(m, 2 * (sum(w[i]*x[i] for i in Iset) +
                        sum(w[i] * uNew[i] for i in IsetC) + tau) >=
                        ((sum(w[i]*lNew[i] for i in Iset) +
                        sum(w[i] * uNew[i] for i in IsetC) + tau) * (1 - yi)) )
    @constraint(m, 2 * sum(w[i]*(uNew[i] - x[i]) for i in Iset) >=
                        ((sum(w[i]*uNew[i] for i in Iset) +
                        sum(w[i]*lNew[i] for i in IsetC) + kappa) * (1 - yi)) )
    return tau, kappa, nonzeroIndices, uNew, lNew
end


function addDenseCons!(m::JuMP.Model, xIn::VarOrAff, xOut::VarOrAff,
                        weights::Array{T, 2},tauList::Array{Float64, 1},
                        kappaList::Array{Float64, 1},
                        nonzeroIndicesList::Array{Array{Int64, 1}},
                        uNewList::Array{Array{Float64, 1}, 1},
                        lNewList::Array{Array{Float64, 1}, 1},
                        cb_data; image=true) where{T<:Real}
    yLen, xLen = length(xOut), length(xIn)
    xVal = zeros(xLen)
    for j in 1:xLen
        xVal[j] = JuMP.callback_value(cb_data, xIn[j])
    end
    contFlag = true
    con1I = -1
    con2I = -1
    con1ValMax = 0
    con2ValMax = 0
    yVal = zeros(yLen)
    for i in 1:yLen
        yVal[i] = aff_callback_value(cb_data, xOut[i])
        if (abs(yVal[i] - 1) < 10^(-10) || abs(yVal[i] + 1) < 10^(-10))
        # if (-1 + 10^(-8) < yVal[i] < 1 - 10^(-8))
            # continue
        else
            contFlag = false
        end
        wVec = weights[i, :]
        nonzeroIndices = nonzeroIndicesList[i]
        nonzeroNum = length(nonzeroIndices)
        if (nonzeroNum == 0)
            continue
        end
        tau, kappa = tauList[i], kappaList[i]
        uNew, lNew = uNewList[i], lNewList[i]
        con1Val, con2Val = decideViolationCons(xVal, yVal[i],nonzeroIndices,
                            wVec, tau, kappa, uNew,lNew)
        if (con1Val > 0.1)
            con1I = i
            wVec = weights[con1I, :]
            nonzeroIndices = nonzeroIndicesList[con1I]
            tau = tauList[con1I]
            uNew, lNew = uNewList[con1I], lNewList[con1I]
            I1 = getFirstCutIndices(xVal, yVal[con1I],nonzeroIndices,wVec,
                                    uNew,lNew)
            con1 = getFirstCon(xIn, xOut[con1I], I1,
                        nonzeroIndices, wVec, tau, uNew, lNew)
            assertFirstCon(xVal, yVal[con1I], I1,
                        nonzeroIndices, wVec, tau, uNew, lNew)
            MOI.submit(m, MOI.UserCut(cb_data), con1)
            m.ext[:CUTS].count += 1
        end
        if (con2Val > 0.1)
            con2I = i
            wVec = weights[con2I, :]
            nonzeroIndices = nonzeroIndicesList[con2I]
            kappa = kappaList[con2I]
            uNew, lNew = uNewList[con2I], lNewList[con2I]
            I2 = getSecondCutIndices(xVal, yVal[con2I],nonzeroIndices,wVec,
                                    uNew,lNew)
            con2 = getSecondCon(xIn, xOut[con2I], I2,
                        nonzeroIndices, wVec, kappa, uNew, lNew)
            assertSecondCon(xVal, yVal[con2I], I2,
                        nonzeroIndices, wVec, kappa, uNew, lNew)
            MOI.submit(m, MOI.UserCut(cb_data), con2)
            m.ext[:CUTS].count += 1
        end
    end

    return contFlag
end


# A transformation for Fourier-Motzkin procedure.
function transformProc(negIndices::Array{Int64, 1},
            upper::Array{T, 1}, lower::Array{U, 1}) where {T<:Real, U<:Real}
    uLen = length(upper)
    upperNew = zeros(uLen)
    lowerNew = zeros(uLen)
    for i in 1:uLen
        if (i in negIndices)
            upperNew[i] = lower[i]
            lowerNew[i] = upper[i]
        else
            upperNew[i] = upper[i]
            lowerNew[i] = lower[i]
        end
    end
    return upperNew, lowerNew
end

function decideViolationCons(xVal::Array{R1, 1}, yVal::R2,
                        nonzeroIndices::Array{Int64, 1},
                        w::Array{R3, 1}, tau::R4, kappa::R5,
                        upper::Array{R6, 1}, lower::Array{R7, 1}; tol = 0
                        ) where {R1<:Real, R2<:Real, R3<:Real, R4<:Real,
                        R5<:Real, R6<:Real, R7<:Real}
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
function getFirstCutIndices(xVal::Array{T, 1}, yVal::U,
                        nonzeroIndices::Array{Int64, 1},
                        w::Array{V, 1},
                        upper::Array{W, 1}, lower::Array{X, 1}
                        ) where {T<:Real, U<:Real, V<:Real, W<:Real, X<:Real}
    I1 = Array{Int64, 1}([])
    for i in nonzeroIndices
        if (2*w[i]*(xVal[i] - upper[i]) < w[i]*(lower[i]-upper[i])*(1-yVal))
            append!(I1, i)
        end
    end
    return I1
end

# Output I^2 for the second constraint in Proposition 1
function getSecondCutIndices(xVal::Array{T, 1}, yVal::U,
                        nonzeroIndices::Array{Int64, 1},
                        w::Array{V, 1},
                        upper::Array{W, 1}, lower::Array{X, 1}
                        ) where {T<:Real, U<:Real, V<:Real, W<:Real, X<:Real}
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
                            w::Array{V, 1},b::T,
                            upper::Array{U, 1}, lower::Array{W, 1}
                            ) where {V<:Real, U<:Real, W<:Real, T <: Real}
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
                            w::Array{V, 1},b::T,
                            upper::Array{U, 1}, lower::Array{W, 1}
                            ) where {V<:Real, U<:Real, W<:Real, T <: Real}
    IsetC = setdiff(nonzeroIndices, Iset)
    return @build_constraint(2 * sum(w[i]*(upper[i] - x[i]) for i in Iset) >=
                        (sum(w[i]*upper[i] for i in Iset) +
                        sum(w[i]*lower[i] for i in IsetC) + b) * (1 - yi)
                        )
end


function assertFirstCon(x, yi,
                            Iset::Array{Int64, 1},
                            nonzeroIndices::Array{Int64, 1},
                            w::Array{V, 1},b::T,
                            upper::Array{U, 1}, lower::Array{W, 1}
                            ) where {V<:Real, U<:Real, W<:Real, T <: Real}
    IsetC = setdiff(nonzeroIndices, Iset)
    # println(2 * (sum(w[Iset] .* x[Iset]) +
    #                     sum(w[IsetC] .* upper[IsetC]) + b) -
    #                     ((sum(w[Iset] .* lower[Iset]) +
    #                     sum(w[IsetC] .* upper[IsetC]) + b) * (1 - yi) ))
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
                            w::Array{V, 1},b::T,
                            upper::Array{U, 1}, lower::Array{W, 1}
                            ) where {V<:Real, U<:Real, W<:Real, T <: Real}
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
