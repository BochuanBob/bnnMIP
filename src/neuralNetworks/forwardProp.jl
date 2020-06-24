function forwardPropBNN(input::Array{Float64}, nn::Array{NNLayer, 1}, varLen::Int64)
    nnLen = length(nn)
    inputLen = length(input)
    output = Array{Float64, 1}(undef, varLen + inputLen + 10)
    count = 1
    xCurrent = input
    for i in 1:nnLen
        if (typeof(nn[i]) == FlattenLayer)
            xCurrent = flatten(xCurrent)
        elseif (typeof(nn[i]) == DenseLayer || typeof(nn[i]) == DenseBinLayer)
            if (typeof(nn[i]) == DenseLayer)
                xCurrent = min.(xCurrent, nn[i].upper)
                xCurrent = max.(xCurrent, nn[i].lower)
                output[(varLen+1):(varLen+inputLen)] = xCurrent
            end
            takeSign = (nn[i].activation == "Sign")
            xCurrent = dense(xCurrent, nn[i].weights,
                        nn[i].bias, takeSign)
            if (takeSign)
                xLen = length(xCurrent)
                output[count:(count+xLen-1)] = (xCurrent .+ 1) / 2
                count += xLen
            end
        elseif (typeof(nn[i]) == Conv2dLayer || typeof(nn[i]) == Conv2dBinLayer)
            xCurrent = conv2d(xCurrent, nn[i].weights,
                        nn[i].bias, nn[i].strides, nn[i].padding)
            xCurrent = signFun.(xCurrent)
            xLen = length(xCurrent)
            output[count:(count+xLen-1)] = (xCurrent[:] .+ 1) / 2
            count += xLen
        else
            error("Not supported layer!")
        end
    end
    output[(varLen + inputLen + 1):(varLen + inputLen + 10)] = xCurrent
    return output
end

function conv2d(x, weights, bias, strides, padding)
    EMPTY3DFLOAT = Array{Float64, 3}(undef, (0,0,0))
    (k1Len, k2Len, channels, filters) = size(weights)
    (s1Len, s2Len) = strides
    (x1Len, x2Len, x3Len) = size(x)
    y3Len = Int64(filters)
    if (padding == "valid")
        y1Len = Int64(floor((x1Len - k1Len) / s1Len) + 1)
        y2Len = Int64(floor((x2Len - k2Len) / s2Len) + 1)
    elseif (padding == "same")
        y1Len = Int64(ceil(x1Len / s1Len))
        y2Len = Int64(ceil(x2Len / s2Len))
    else
        error("Not supported padding!")
    end
    outputSize = (y1Len, y2Len, y3Len)
    y = zeros(outputSize)
    for i in 1:y1Len
        for j in 1:y2Len
            for k in 1:y3Len
                weight = EMPTY3DFLOAT
                x1Start, x1End = 1 + (i-1)*s1Len, (i-1)*s1Len + k1Len
                x2Start, x2End = 1 + (j-1)*s2Len, (j-1)*s2Len + k2Len
                xN = EMPTY3DFLOAT
                if (x1Start <= x1Len && x2Start <= x2Len)
                    xN = x[x1Start:min(x1Len, x1End),
                            x2Start:min(x2Len, x2End), :]
                    (xN1Len, xN2Len, _) = size(xN)
                    weight = weights[1:xN1Len, 1:xN2Len, :, k]
                end
                y[i,j,k] = sum(xN .* weight) + bias[k]
            end
        end
    end
    return y
end

function dense(xCurrent, weight, bias, takeSign)
    if (takeSign)
        xCurrent = signFun.(weight * xCurrent + bias)
    else
        xCurrent = weight * xCurrent + bias
    end
    return xCurrent
end

function flatten(x)
    dim = length(size(x))
    return permutedims(x, Array{Int64,1}(dim:-1:1))[:]
end
