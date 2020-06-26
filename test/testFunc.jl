using Random
export forwardProp, signFun, forwardPropInt, keepOnlyKEntries, keepOnlyKEntriesSeq
# Keep random k entries per row in a matrix and set all the others to 0.
function keepOnlyKEntries(mat, k::Int64)
    (xLen, yLen) = size(mat)
    for i in 1:xLen
        arr = findall(mat[i,:] .!= 0)
        arrLen = length(arr)
        if (arrLen <= k)
            continue
        end
        jList = randperm(arrLen)[1:(arrLen-k)]
        for j in jList
            mat[i,arr[j]] = 0
        end
    end
    return mat
end

function keepOnlyKEntriesSeq(mat, k::Int64)
    (xLen, yLen) = size(mat)
    for i in 1:xLen
        jList = setdiff(1:yLen, mod.((1:k) .+floor(k/2) * (i-1), yLen))
        for j in jList
            mat[i, j] = 0
        end
    end
    return mat
end

function forwardProp(image::Array, nn::Array{NNLayer, 1})
    xCurrent = image
    for i in 1:length(nn)
        if (typeof(nn[i]) == FlattenLayer)
            xCurrent = flatten(xCurrent)
        elseif (typeof(nn[i]) == DenseLayer || typeof(nn[i]) == DenseBinLayer)
            takeSign = (nn[i].activation == "Sign")
            xCurrent = dense(xCurrent, nn[i].weights,
                        nn[i].bias, takeSign)
        elseif (typeof(nn[i]) == Conv2dLayer || typeof(nn[i]) == Conv2dBinLayer)
            xCurrent = conv2d(xCurrent, nn[i].weights,
                        nn[i].bias, nn[i].strides, nn[i].padding)
            xCurrent = signFun.(xCurrent)
        else
            error("Not supported layer!")
        end
    end
    return xCurrent
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

function forwardPropInt(image::Array)
    image = permutedims(image, [3,2,1])[:]
    weight1 = nn[2]["weights"]
    bias1 = nn[2]["bias"] * 255
    weight2 = nn[3]["weights"]
    bias2 = nn[3]["bias"]
    weight3 = nn[4]["weights"]
    bias3 = nn[4]["bias"]

    l1 = signFun.(weight1 * image + bias1)
    l2 = signFun.(weight2 * l1 + bias2)
    l3 = weight3 * l2 + bias3
    return l3
end


function signFun(x::T) where {T<:Real}
    if (x >=0)
        return 1.0
    else
        return -1.0
    end
end
