include("layerSetup.jl")
include("activation.jl")
include("denseSetup.jl")
export dense, getDenseCons

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
    if (actFunc=="Sign")
        z = @variable(m, [1:yLen], binary=true,
                    base_name="z_$count")
        @constraint(m, [i=1:yLen], y[i] == 2 * z[i] - 1)
        for i in 1:yLen
            neuronSign!(m, x, y[i], weights[i, :], bias[i],
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
    return y
end

# A MIP formulation for a single neuron.
function neuronSign!(m::JuMP.Model, x::VarOrAff, yi::VarOrAff,
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
        return nothing
    end
    if (image)
        tau, kappa = getTauAndKappa(nonzeroNum, 255*b)
        tau = tau/255
        kappa = kappa/255
    else
        tau, kappa= b, b
    end
    uNew, lNew = transformProc(negIndices, upper, lower)
    Iset1 = union(posIndices, negIndices)
    @constraint(m, getBNNCutFirstConGE(m, x, yi, Iset1,
                nonzeroIndices,wVec,tau,uNew, lNew)>=0)
    @constraint(m, getBNNCutSecondConGE(m, x, yi, Iset1,
                nonzeroIndices,wVec,kappa,uNew, lNew)>=0)
    Iset2 = Array{Int64, 1}([])
    @constraint(m, getBNNCutFirstConGE(m, x, yi, Iset2,
                nonzeroIndices,wVec,tau,uNew, lNew)>=0)
    @constraint(m, getBNNCutSecondConGE(m, x, yi, Iset2,
                nonzeroIndices,wVec,kappa,uNew, lNew)>=0)
    return nothing
end

function getDenseCons(m::JuMP.Model, xIn::VarOrAff, xOut::VarOrAff,
                        weights::Array{T, 2},bias::Array{U, 1},
                        upper::Array{V, 1}, lower::Array{W, 1},
                        cb_data; image=true) where{T<:Real, U<:Real, V<:Real, W<:Real}
    conList = []
    (yLen, xLen) = size(weights)
    xVal = zeros(length(xIn))
    for j in 1:xLen
        xVal[j] = JuMP.callback_value(cb_data, xIn[j])
    end
    for i in 1:yLen
        wVec = weights[i, :]
        b = bias[i]
        posIndices = findall(wVec .> 0)
        negIndices = findall(wVec .< 0)
        nonzeroIndices = union(posIndices, negIndices)
        nonzeroNum = length(nonzeroIndices)
        if (image)
            tau, kappa = getTauAndKappa(nonzeroNum, 255*b)
            tau = tau/255
            kappa = kappa/255
        else
            tau, kappa= b, b
        end
        uNew, lNew = transformProc(negIndices, upper, lower)
        if (nonzeroNum == 0)
            continue
        end
        yVal = JuMP.callback_value(cb_data, xOut[i])
        I1, I2 = getCutsIndices(xVal, yVal,nonzeroIndices,wVec,
                                uNew,lNew)
        con1 = @build_constraint(getBNNCutFirstConGE(m, xIn, xOut[i], I1,
                    nonzeroIndices,wVec,tau,uNew, lNew)>=0)
        con2 = @build_constraint(getBNNCutSecondConGE(m, xIn, xOut[i], I2,
                    nonzeroIndices,wVec,kappa,uNew, lNew)>=0)
        conList = vcat(conList, con1)
        conList = vcat(conList, con2)
    end
    return conList
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
    # TODO: Implement the function.
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
    # TODO: Implement the function.
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
    # TODO: Implement the function.
    IsetC = setdiff(nonzeroIndices, Iset)
    expr = @expression(m, 2 * sum(w[i]*(upper[i] - x[i]) for i in Iset) -
                        (sum(w[i]*upper[i] for i in Iset) +
                        sum(w[i]*lower[i] for i in IsetC) + b) * (1 - yi)
                        )
    return expr
end
