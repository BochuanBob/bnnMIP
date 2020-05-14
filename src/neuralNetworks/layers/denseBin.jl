include("layerSetup.jl")
export denseBin

# A fully connected layer with sign() or
# without activation function.
# Each entry of weights must be -1, 0, 1.
# xOnes: xOne is true if each entry of input is either -1 or 1.
# False otherwise.
function denseBin(m::JuMP.Model, x::VarOrAff,
               weights::Array{T, 2}, bias::Array{U, 1};
               takeSign=false, cuts=true, xOnes=true) where{T<:Real, U<:Real}
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
    y = nothing
    if (takeSign)
        y = @variable(m, [1:yLen],
                    base_name="y_$count")
        z = @variable(m, [1:yLen], binary=true,
                    base_name="z_$count")
        @constraint(m, [i=1:yLen], y[i] == 2 * z[i] - 1)
        for i in 1:yLen
            neuronSign!(m, x, y[i], weights[i, :], bias[i], cuts=cuts,
                        xOnes=xOnes)
        end
    else
        y = @variable(m, [1:yLen],
                    base_name="y_$count")
        @constraint(m, [i=1:yLen], y[i] ==
                    bias[i] + sum(weights[i,j] * x[j] for j in 1:xLen))
    end
    return y
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
# If cuts == false, it is a Big-M formulation.
# Otherwise, cuts for the ideal formulation are added to the model.
function neuronSign!(m::JuMP.Model, x::VarOrAff, yi::VarOrAff,
                weightVec::Array{T, 1}, b::U;
                cuts=false, xOnes=true) where{T<:Real, U<:Real}
    # initNN!(m)
    oneIndices = findall(weightVec .== 1)
    negOneIndices = findall(weightVec .== -1)
    nonzeroNum = length(oneIndices) + length(negOneIndices)
    if (nonzeroNum == 0)
        @constraint(m, yi == 1)
        return nothing
    end
    tau = b
    kappa = b
    if (xOnes)
        tau, kappa = getTauAndKappa(nonzeroNum, b)
    end
    Iset1 = union(oneIndices, negOneIndices)
    @constraint(m, getBNNCutFirstConGE(m, x, yi, Iset1,
                oneIndices, negOneIndices, tau)>=0)
    @constraint(m, getBNNCutSecondConLE(m, x, yi, Iset1,
                oneIndices, negOneIndices, kappa)<=0)
    Iset2 = Array{Int64, 1}([])
    @constraint(m, getBNNCutFirstConGE(m, x, yi, Iset2,
                    oneIndices, negOneIndices, tau)>=0)
    @constraint(m, getBNNCutSecondConLE(m, x, yi, Iset2,
                    oneIndices, negOneIndices, kappa)<=0)
    if (cuts)
        # Generate cuts by callback function
        function callbackCutsBNN(cb_data)
            xVal = callback_value(cb_data, x)
            yVal = callback_value(cb_data, yi)
            I1, I2 = getCutsIndices(xVal, yVal,oneIndices,negOneIndices)
            con1 = @build_constraint(getBNNCutFirstConGE(m, x, yi, I1,
                            oneIndices, negOneIndices, tau)>=0)
            con2 = @build_constraint(getBNNCutSecondConLE(m, x, yi, I2,
                            oneIndices, negOneIndices, kappa) <= 0)
            MOI.submit(m, MOI.UserCut(cb_data), con1)
            MOI.submit(m, MOI.UserCut(cb_data), con2)
        end
        MOI.set(m, MOI.UserCutCallback(), callbackCutsBNN)
    end
    return nothing
end

# Output the I^1 and I^2 for two constraints in Proposition 3
function getCutsIndices(xVal::Array{T, 1}, yVal::U,
                        oneIndices::Array{Int64, 1},
                        negOneIndices::Array{Int64, 1}) where {T<:Real, U<:Real}
    xLen = length(xVal)
    I1 = Array{Int64, 1}([])
    I2 = Array{Int64, 1}([])
    for i in oneIndices
        if (xVal[i] < yVal)
            append!(I1, i)
        elseif (xVal[i] > yVal)
            append!(I2, i)
        end
    end

    for i in negOneIndices
        if (-xVal[i] < yVal)
            append!(I1, i)
        elseif (-xVal[i] > yVal)
            append!(I2, i)
        end
    end

    return I1, I2
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

# When w^T x + kappa <= 0, sign(w^T x + b) = -1.
# When w^T x + tau >= 0, sign(w^T x + b) = 1.
# Each entry of w must be in -1, 0, 1.
# Return: tau, kappa.
function getTauAndKappa(nonzeroNum::Int64, b::U) where {U<:Real}
    if (floor(b) == ceil(b))
        return getTauAndKappaInt(nonzeroNum, Int64(b))
    else
        return getTauAndKappaFloat(nonzeroNum, Float64(b))
    end
end

# Get tau and kappa when b is an integer.
# Return: tau, kappa.
function getTauAndKappaInt(nonzeroNum::Int64, b::Int64)
    tau = Inf
    kappa = Inf
    if (mod(b, 2) == 0)
        if(mod(nonzeroNum, 2) == 0)
            tau = b
            kappa = b + 2
        else
            tau = b - 1
            kappa = b + 1
        end
    else
        if(mod(nonzeroNum, 2) == 0)
            tau = b - 1
            kappa = b + 1
        else
            tau = b
            kappa = b + 2
        end
    end
    return Int64(tau), Int64(kappa)
end

# Get tau and kappa when b is a real number but not an integer.
# Return: tau, kappa.
function getTauAndKappaFloat(nonzeroNum::Int64, b::Float64)
    tau = Inf
    kappa = Inf
    if (floor(b) == ceil(b))
        error("b can not be an integer!")
    end
    if (mod(floor(b), 2) == 0)
        if(mod(nonzeroNum, 2) == 0)
            tau = floor(b)
            kappa = ceil(b) + 1
        else
            tau = floor(b) - 1
            kappa = ceil(b)
        end
    else
        if(mod(nonzeroNum, 2) == 0)
            tau = floor(b) - 1
            kappa = ceil(b)
        else
            tau = floor(b)
            kappa = ceil(b) + 1
        end
    end
    return Int64(tau), Int64(kappa)
end
