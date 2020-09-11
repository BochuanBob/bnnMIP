export denseLP

function denseLP(m::JuMP.Model, x::Array{VariableRef},
                     weights::Array{Float64, 2}, bias::Array{Float64, 1},
                     upper::Array{Float64, 1}, lower::Array{Float64, 1};
                     actFunc="", actBits=1)
    initNN!(m)
    count = m.ext[:NN].count
    m.ext[:NN].count += 1
    (yLen, xLen) = size(weights)
    if (length(bias) != yLen)
        error("The sizes of weights and bias don't match!")
    end

    @constraint(m, x .<= upper)
    @constraint(m, x .>= lower)

    upperOut = zeros(yLen)
    lowerOut = zeros(yLen)
    for i in 1:yLen
        upperOut[i], lowerOut[i] = calcBounds(upper, lower, weights[i, :],
                                            bias[i])
    end
    v = weights * x .+ bias
    if (actFunc == "DoReFa")
        y = @variable(m, [1:yLen],
                    base_name="y_$count")
        @constraint(m, [i=1:yLen], y[i] <= DoReFa(upperOut[i], actBits))
        @constraint(m, [i=1:yLen], y[i] >= DoReFa(lowerOut[i], actBits))
        for i in 1:yLen
            addLPcons!(m, v[i], y[i], upperOut[i], lowerOut[i], Int64.(actBits))
        end
    elseif (actFunc == "")
        y = @variable(m, [1:yLen],
                    base_name="y_$count")
        @constraint(m, [i=1:yLen], y[i] ==
                    bias[i] + sum(weights[i,j] * x[j] for j in 1:xLen))
    else
        error("Not supported activation functions for dense layer.")
    end
    return y, upperOut, lowerOut
end

function calcBounds(upper::Array{Float64, 1}, lower::Array{Float64, 1},
                    weights::Array{Float64, 1}, bias::Float64)
    upperOut, lowerOut = 0, 0
    wLen = length(weights)
    for i in 1:wLen
        if (weights[i] > 0)
            upperOut += upper[i] * weights[i]
            lowerOut += lower[i] * weights[i]
        else
            upperOut += lower[i] * weights[i]
            lowerOut += upper[i] * weights[i]
        end
    end
    return upperOut+bias, lowerOut+bias
end

function DoReFa(input, actBits)
    output = max(0, input)
    output = min(output, 1)
    n = 2^actBits - 1
    output = round(output * n) / n
    return output
end

function addLPcons!(m::JuMP.Model, v::VarOrAff, y::VarOrAff,
                    upper::Float64, lower::Float64, actBits::Int64)
    n = 2^actBits - 1
    qL = DoReFa(lower, actBits)
    qU = DoReFa(upper, actBits)
    B1 = (2 * floor(n * lower + 1/2) + 1) / (2 * n)
    B2 = (2 * floor(n * upper + 1/2) - 1) / (2 * n)
    if (upper >= (2*n+1) / (2 * n))
        @constraint(m, y >= (v - B1) * (1 - qL) / (upper - B1) + qL)
    else
        @constraint(m, y >= v - 1 / (2*n))
        @constraint(m, y >= (v - B2) / (n * (upper - B2)) + qU - 1/n )
    end

    if (lower <= -1/(2 * n))
        @constraint(m, y <= qU * (v - lower) / (B2 - lower))
    else
        @constraint(m, y <= v + 1 / (2 * n))
        @constraint(m, y <= (v - lower)/ (n * (B1 - lower)) + qL)
    end
    return nothing
end
