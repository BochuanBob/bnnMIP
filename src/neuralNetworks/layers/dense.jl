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
    y = @variable(m, [1:yLen],
                base_name="y_$count")
    tauList = zeros(yLen)
    kappaList = zeros(yLen)
    nonzeroIndicesList = Array{Array{Int64, 1}, 1}(undef, yLen)
    uNewList = Array{Array{Float64, 1}, 1}(undef, yLen)
    lNewList = Array{Array{Float64, 1}, 1}(undef, yLen)
    if (actFunc=="Sign")
        z = @variable(m, [1:yLen], binary=true,
                    base_name="z_$count")
        @constraint(m, [i=1:yLen], y[i] == 2 * z[i] - 1)
        for i in 1:yLen
            tauList[i], kappaList[i], nonzeroIndicesList[i],
                uNewList[i], lNewList[i] = neuronDenseSign(m, x, y[i],
                weights[i, :], bias[i],
                upper, lower, image=image)
        end
    else
        M = 1000
        z = @variable(m, [1:yLen],
                    base_name="z_$count")
        @constraint(m, [i=1:yLen], z[i] ==
                    bias[i] + sum(weights[i,j] * x[j] for j in 1:xLen))
        # TODO: Determine the values of lower and upper bounds.
        y = activation1D(m, z, actFunc, upper=M, lower=-M)
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
    Iset1 = union(posIndices, negIndices)
    @constraint(m, getBNNCutFirstConGE(m, x, yi, Iset1,
                nonzeroIndices,wVec,tau,uNew, lNew)>=0)
    @constraint(m, getBNNCutSecondConGE(m, x, yi, Iset1,
                nonzeroIndices,wVec,kappa,uNew, lNew)>=0)
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
    for i in 1:yLen
        yVal = JuMP.callback_value(cb_data, xOut[i])
        if (abs(yVal - 1) < 10^(-4) || abs(yVal + 1) < 10^(-4))
            continue
        else
            # contFlag = false
        end
        wVec = weights[i, :]
        nonzeroIndices = nonzeroIndicesList[i]
        nonzeroNum = length(nonzeroIndices)
        if (nonzeroNum == 0)
            continue
        end
        tau, kappa = tauList[i], kappaList[i]
        uNew, lNew = uNewList[i], lNewList[i]
        I1, I2 = getCutsIndices(xVal, yVal,nonzeroIndices,wVec,
                                uNew,lNew)
        try
            con1 = getFirstCon(xIn, xOut[i], I1,
                        nonzeroIndices, wVec, tau, uNew, lNew)
            con2 = getSecondCon(xIn, xOut[i], I2,
                        nonzeroIndices, wVec, kappa, uNew, lNew)
            MOI.submit(m, MOI.UserCut(cb_data), con1)
            MOI.submit(m, MOI.UserCut(cb_data), con2)
        catch e
        end
    end
    return contFlag
end


# A transformation for Fourier-Motzkin procedure.
# When
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

# Output the I^1 and I^2 for two constraints in Proposition 1
function getCutsIndices(xVal::Array{T, 1}, yVal::U,
                        nonzeroIndices::Array{Int64, 1},
                        w::Array{V, 1},
                        upper::Array{W, 1}, lower::Array{X, 1}
                        ) where {T<:Real, U<:Real, V<:Real, W<:Real, X<:Real}
    I1 = Array{Int64, 1}([])
    I2 = Array{Int64, 1}([])
    for i in nonzeroIndices
        if (2*w[i]*(xVal[i] - upper[i]) < w[i]*(lower[i]-upper[i])*(1-yVal))
            append!(I1, i)
        end
        if (2*w[i]*(upper[i]-xVal[i]) < w[i]*(upper[i]-lower[i])*(1-yVal))
            append!(I2, i)
        end
    end
    return I1, I2
end

# Return first constraint with given I, I^+, I^-, tau, kappa as shown
# in Proposition 1.
function getBNNCutFirstConGE(m::JuMP.Model,
                            x::VarOrAff, yi::VarOrAff,
                            Iset::Array{Int64, 1},
                            nonzeroIndices::Array{Int64, 1},
                            w::Array{V, 1},b::T,
                            upper::Array{U, 1}, lower::Array{W, 1}
                            ) where {V<:Real, U<:Real, W<:Real, T <: Real}
    IsetC = setdiff(nonzeroIndices, Iset)
    expr = @expression(m, 2 * (sum(w[i]*x[i] for i in Iset) +
                        sum(w[i] * upper[i] for i in IsetC) + b) -
                        (sum(w[i]*lower[i] for i in Iset) +
                        sum(w[i] * upper[i] for i in IsetC) + b) * (1 - yi)
                        )
    return expr
end

# Return second constraint with given I, I^+, I^-, tau, kappa as shown
# in Proposition 1.
function getBNNCutSecondConGE(m::JuMP.Model,
                            x::VarOrAff, yi::VarOrAff,
                            Iset::Array{Int64, 1},
                            nonzeroIndices::Array{Int64, 1},
                            w::Array{V, 1},b::T,
                            upper::Array{U, 1}, lower::Array{W, 1}
                            ) where {V<:Real, U<:Real, W<:Real, T <: Real}
    IsetC = setdiff(nonzeroIndices, Iset)
    expr = @expression(m, 2 * sum(w[i]*(upper[i] - x[i]) for i in Iset) -
                        (sum(w[i]*upper[i] for i in Iset) +
                        sum(w[i]*lower[i] for i in IsetC) + b) * (1 - yi)
                        )
    return expr
end


# Return first constraint with given I, I^+, I^-, tau, kappa as shown
# in Proposition 1.
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
# in Proposition 1.
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
