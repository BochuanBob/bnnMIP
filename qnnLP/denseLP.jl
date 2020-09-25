export denseLP

function denseLP(m::JuMP.Model, x::Array{VariableRef},
                     weights::Array{Float64, 2}, bias::Array{Float64, 1},
                     upper::Array{Float64, 1}, lower::Array{Float64, 1};
                     actFunc="", actBits=1, MIP=false)
    initNN!(m)
    count = m.ext[:NN].count
    m.ext[:NN].count += 1
    (yLen, xLen) = size(weights)
    if (length(bias) != yLen)
        println(size(bias))
        println(size(weights))
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
    y = @variable(m, [1:yLen],
                base_name="y_$count")
    if (actFunc == "DoReFa")
        if (MIP)
            # z = @variable(m, [1:yLen, 1:(2^actBits)], base_name="z_$count")
            for i in 1:yLen
                # addDoReFaMIPcons!(m, v[i], y[i], z[i, :], upperOut[i],
                #                 lowerOut[i], Int64.(actBits))
                upperOut[i] = DoReFa(upperOut[i], actBits)
                lowerOut[i] = DoReFa(lowerOut[i], actBits)
            end
        else
            @constraint(m, [i=1:yLen], y[i] <= DoReFa(upperOut[i], actBits))
            @constraint(m, [i=1:yLen], y[i] >= DoReFa(lowerOut[i], actBits))
            for i in 1:yLen
                addDoReFaLPcons!(m, v[i], y[i], upperOut[i], lowerOut[i], Int64.(actBits))
                upperOut[i] = DoReFa(upperOut[i], actBits)
                lowerOut[i] = DoReFa(lowerOut[i], actBits)
            end
        end
    elseif (actFunc == "Sign")
        if (MIP)
            z = @variable(m, [1:yLen], binary=true,
                        base_name="z_$count")
            @constraint(m, 2 .* z .- 1 .== y)
            for i in 1:yLen
                @constraint(m, 2*v >= lowerOut[i] * (1 - y[i]))
                @constraint(m, 2*v <= upperOut[i] * (1 + y[i]))
                upperOut[i] = Sign(upperOut[i])
                lowerOut[i] = Sign(lowerOut[i])
            end
        else
            @constraint(m, [i=1:yLen], y[i] <= Sign(upperOut[i]))
            @constraint(m, [i=1:yLen], y[i] >= Sign(lowerOut[i]))
            for i in 1:yLen
                addSignLPcons!(m, v[i], y[i], upperOut[i], lowerOut[i])
                upperOut[i] = Sign(upperOut[i])
                lowerOut[i] = Sign(lowerOut[i])
            end
        end
    elseif (actFunc == "")
        @constraint(m, [i=1:yLen], y[i] ==
                    bias[i] + sum(weights[i,j] * x[j] for j in 1:xLen))
    else
        error("Not supported activation functions for dense layer.")
    end
    return y, upperOut, lowerOut
end

function calcBounds(upper::Array{Float64, 1}, lower::Array{Float64, 1},
                    weights::Array{Float64, 1}, bias::Float64)
    upperOut, lowerOut = bias, bias
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
    return upperOut, lowerOut
end

function Sign(input)
    if (input >= 0)
        return 1
    else
        return -1
    end
end

function DoReFa(input, actBits)
    output = max(0, input)
    output = min(output, 1)
    n = 2^actBits - 1
    output = round(output * n) / n
    return output
end

function addSignLPcons!(m::JuMP.Model, v::VarOrAff, y::VarOrAff,
                    upper::Float64, lower::Float64)
    if (Sign(upper) == Sign(lower))
        return nothing
    end
    @constraint(m, y >= (2 / upper) * v - 1)
    @constraint(m, y <= -(2 / lower) * v + 1)
    return nothing
end

function addDoReFaLPcons!(m::JuMP.Model, v::VarOrAff, y::VarOrAff,
                    upper::Float64, lower::Float64, actBits::Int64)
    n = 2^actBits - 1
    qL = DoReFa(lower, actBits)
    qU = DoReFa(upper, actBits)
    if (qL == qU)
        return nothing
    end
    B1 = (2 * floor(n * lower + 1/2) + 1) / (2 * n)
    B1 = min(max(B1, 0), 1)
    B2 = (2 * floor(n * upper + 1/2) - 1) / (2 * n)
    B2 = min(max(B2, 0), 1)
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

function addDoReFaMIPcons!(m::JuMP.Model, v::VarOrAff, y::VarOrAff,
                    z::Array{VariableRef},
                    upper::Float64, lower::Float64, actBits::Int64)
    # TODO
end
