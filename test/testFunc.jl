export forwardProp, signFun
function forwardProp(image::Array)
    image = permutedims(image, [3,2,1])[:]
    weight1 = nn[2]["weights"]
    bias1 = nn[2]["bias"]
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
