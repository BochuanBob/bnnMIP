include("layerSetup.jl")
include("activation.jl")
export dense

# The MIP formulation for general dense layer
function dense(m::JuMP.Model, x::VarOrAff,
               weights::Array{T, 2}, bias::Array{U, 1},
               upper::Array{V, 1}, lower::Array{W, 1};
               actFunc="", cuts=true
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
                        upper, lower, cuts=cuts)
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
# If cuts == false, it is a Big-M formulation.
# Otherwise, cuts for the ideal formulation are added to the model.
function neuronSign!(m::JuMP.Model, x::VarOrAff, yi::VarOrAff,
                weightVec::Array{T, 1}, b::U,
                upper::Array{V, 1}, lower::Array{W, 1};
                cuts=false) where{T<:Real, U<:Real, V<:Real, W<:Real}
    # initNN!(m)
    posIndices = findall(weightVec .> 0)
    negIndices = findall(weightVec .< 0)
    nonzeroIndices = union(posIndices, negIndices)
    nonzeroNum = length(nonzeroIndices)
    if (nonzeroNum == 0)
        @constraint(m, yi == 1)
        return nothing
    end
    Iset1 = union(posIndices, negIndices)
    xNew, uNew, lNew = transformProc(m, negIndices, x, upper, lower)
    wVec=abs.(weightVec)
    @constraint(m, getBNNCutFirstConGE(m, xNew, yi, Iset1,
                nonzeroIndices,wVec,b,uNew, lNew)>=0)
    @constraint(m, getBNNCutSecondConGE(m, xNew, yi, Iset1,
                nonzeroIndices,wVec,b,uNew, lNew)>=0)
    Iset2 = Array{Int64, 1}([])
    @constraint(m, getBNNCutFirstConGE(m, xNew, yi, Iset2,
                nonzeroIndices,wVec,b,uNew, lNew)>=0)
    @constraint(m, getBNNCutSecondConGE(m, xNew, yi, Iset2,
                nonzeroIndices,wVec,b,uNew, lNew)>=0)
    if (cuts)
        # Generate cuts by callback function
        function callbackCutsBNN(cb_data)
            xLen = length(xNew)
            xVal = zeros(length(xNew))
            for i in 1:xLen
                xVal[i] = JuMP.callback_value(cb_data, xNew[i])
            end
            yVal = JuMP.callback_value(cb_data, yi)
            I1, I2 = getCutsIndices(xVal, yVal,nonzeroIndices,wVec,
                                    upper,lower)
            con1 = @build_constraint(getBNNCutFirstConGE(m, xNew, yi, I1,
                        nonzeroIndices,wVec,b,uNew, lNew)>=0)
            con2 = @build_constraint(getBNNCutSecondConGE(m, xNew, yi, I2,
                        nonzeroIndices,wVec,b,uNew, lNew)>=0)
            MOI.submit(m, MOI.UserCut(cb_data), con1)
            MOI.submit(m, MOI.UserCut(cb_data), con2)
        end
        MOI.set(m, MOI.UserCutCallback(), callbackCutsBNN)
    end
    return nothing
end
# A transformation for Fourier-Motzkin procedure.
# When
function transformProc(m::JuMP.Model, negIndices::Array{Int64, 1}, x::VarOrAff,
            upper::Array{T, 1}, lower::Array{U, 1}) where {T<:Real, U<:Real}
    initNN!(m)
    count = m.ext[:NN].count
    xLen = length(x)
    upperNew = zeros(xLen)
    lowerNew = zeros(xLen)
    # z = @variable(m, [1:xLen], base_name="xNew_$count")
    z = @variable(m, [1:xLen])
    for i in 1:xLen
        if (i in negIndices)
            @constraint(m, z[i] == -x[i])
            upperNew[i] = -lower[i]
            lowerNew[i] = -upper[i]
        else
            @constraint(m, z[i] == x[i])
            upperNew[i] = upper[i]
            lowerNew[i] = lower[i]
        end
    end
    return z, upperNew, lowerNew
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
