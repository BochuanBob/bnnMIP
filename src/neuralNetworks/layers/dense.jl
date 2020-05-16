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
    if (actFun="Sign")
        z = @variable(m, [1:yLen], binary=true,
                    base_name="z_$count")
        @constraint(m, [i=1:yLen], y[i] == 2 * z[i] - 1)
        for i in 1:yLen
            neuronSign!(m, x, y[i], weights[i, :], bias[i], cuts=cuts)
        end
    else
        M = 1000
        z = @variable(m, [1:yLen],
                    base_name="z_$count")
        @constraint(m, [i=1:yLen], z[i] ==
                    bias[i] + sum(weights[i,j] * x[j] for j in 1:xLen))
        # TODO: Determine the values of lower and upper bounds.
        y = activation1D(m, z, actFun, upper=M, lower=-M)
    end
    return y
end

# A MIP formulation for a single neuron.
# If cuts == false, it is a Big-M formulation.
# Otherwise, cuts for the ideal formulation are added to the model.
function neuronSign!(m::JuMP.Model, x::VarOrAff, yi::VarOrAff,
                weightVec::Array{T, 1}, b::U;
                cuts=false) where{T<:Real, U<:Real}
    # initNN!(m)
    oneIndices = findall(weightVec .> 0)
    negOneIndices = findall(weightVec .< -1)
    nonzeroNum = length(oneIndices) + length(negOneIndices)
    if (nonzeroNum == 0)
        @constraint(m, yi == 1)
        return nothing
    end
    Iset1 = union(oneIndices, negOneIndices)
    @constraint(m, getBNNCutFirstConGE(m, x, yi, Iset1,
                oneIndices, negOneIndices,weightVec, b)>=0)
    @constraint(m, getBNNCutSecondConLE(m, x, yi, Iset1,
                oneIndices, negOneIndices,weightVec, b)<=0)
    Iset2 = Array{Int64, 1}([])
    @constraint(m, getBNNCutFirstConGE(m, x, yi, Iset2,
                    oneIndices, negOneIndices,weightVec, b)>=0)
    @constraint(m, getBNNCutSecondConLE(m, x, yi, Iset2,
                    oneIndices, negOneIndices,weightVec, b)<=0)
    if (cuts)
        # Generate cuts by callback function
        function callbackCutsBNN(cb_data)
            xVal = callback_value(cb_data, x)
            yVal = callback_value(cb_data, yi)
            I1, I2 = getCutsIndices(xVal, yVal,oneIndices,negOneIndices)
            con1 = @build_constraint(getBNNCutFirstConGE(m, x, yi, I1,
                            oneIndices, negOneIndices,weightVec, b)>=0)
            con2 = @build_constraint(getBNNCutSecondConLE(m, x, yi, I2,
                            oneIndices, negOneIndices,weightVec, b) <= 0)
            MOI.submit(m, MOI.UserCut(cb_data), con1)
            MOI.submit(m, MOI.UserCut(cb_data), con2)
        end
        MOI.set(m, MOI.UserCutCallback(), callbackCutsBNN)
    end
    return nothing
end

# Output the I^1 and I^2 for two constraints in Proposition 1
function getCutsIndices(xVal::Array{T, 1}, yVal::U,
                        oneIndices::Array{Int64, 1},
                        negOneIndices::Array{Int64, 1},
                        weightvec::Array{V, 2}
                        ) where {T<:Real, U<:Real, V<:Real}
    # TODO: Implement the function.
    return nothing
end

# Return first constraint with given I, I^+, I^-, tau, kappa as shown
# in Proposition 1.
function getBNNCutFirstConGE(m::JuMP.Model,
                            x::VarOrAff, yi::VarOrAff,
                            Iset::Array{Int64, 1},
                            oneIndices::Array{Int64, 1},
                            negOneIndices::Array{Int64, 1},
                            tau::T) where {T <: Real}
    # TODO: Implement the function.
    return nothing
end

# Return second constraint with given I, I^+, I^-, tau, kappa as shown
# in Proposition 1.
function getBNNCutSecondConLE(m::JuMP.Model,
                            x::VarOrAff, yi::VarOrAff,
                            Iset::Array{Int64, 1},
                            oneIndices::Array{Int64, 1},
                            negOneIndices::Array{Int64, 1},
                            kappa::T) where {T <: Real}
    # TODO: Implement the function.
    return expr
end
