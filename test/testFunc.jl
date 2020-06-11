export forwardProp, signFun, forwardPropInt, keepOnlyKEntries, keepOnlyKEntriesSeq
using Random
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

function forwardProp(image::Array, nn)
    image = permutedims(image, [3,2,1])[:]
    xCurrent = image
    for i in 2:length(nn)
        weight = nn[i]["weights"]
        bias = nn[i]["bias"]
        takeSign = false
        if (haskey(nn[i], "activation"))
            takeSign = (nn[i]["activation"] == "Sign")
        end
        if (takeSign)
            xCurrent = signFun.(weight * xCurrent + bias)
        else
            xCurrent = weight * xCurrent + bias
        end
    end
    return xCurrent
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
