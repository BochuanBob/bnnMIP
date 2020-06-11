include("layerSetup.jl")
include("activation.jl")
include("denseSetup.jl")
export conv2dSign, addDenseCons!

const EMPTY3DVAR = Array{VariableRef, 3}(undef, (0,0,0))
const EMPTY3DFLOAT = Array{Float64, 3}(undef, (0,0,0))

# The MIP formulation for general conv2d layer
function conv2dSign(m::JuMP.Model, x::VarOrAff,
               weights::Array{T, 4}, bias::Array{U, 1},
               strides::Tuple{Int64, Int64},
               upper::Array{V, 3}, lower::Array{W, 3};
               padding="valid", image=true, preCut=true
               ) where{T<:Real, U<:Real, V<:Real, W<:Real}
    initNN!(m)
    count = m.ext[:NN].count
    m.ext[:NN].count += 1
    (k1Len, k2Len, channels, filters) = size(weights)
    (s1Len, s2Len) = strides
    (x1Len, x2Len, x3Len) = size(x)
    if (length(bias) != filters)
        error("The sizes of weights and bias don't match!")
    end
    if (x3Len != channels)
        error("The length of channels doesn't match!")
    end
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

    @constraint(m, x .<= upper)
    @constraint(m, x .>= lower)

    outputSize = (y1Len, y2Len, y3Len)
    tauList = zeros(outputSize)
    kappaList = zeros(outputSize)
    nonzeroIndicesList = Array{Array{CartesianIndex{3}, 1}, 3}(undef, outputSize)
    uNewList = Array{Array{Float64, 3}, 3}(undef, outputSize)
    lNewList = Array{Array{Float64, 3}, 3}(undef, outputSize)
    z = @variable(m, [1:y1Len, 1:y2Len, 1:y3Len], binary=true,
                base_name="z_$count")
    y = @expression(m, 2 .* z .- 1)
    println("y: ", size(y))
    for i in 1:y1Len
        for j in 1:y2Len
            for k in 1:y3Len
                weight = EMPTY3DFLOAT
                x1Start, x1End = 1 + (i-1)*s1Len, (i-1)*s1Len + k1Len
                x2Start, x2End = 1 + (j-1)*s2Len, (j-1)*s2Len + k2Len
                xN = EMPTY3DVAR
                ub = EMPTY3DFLOAT
                lb = EMPTY3DFLOAT
                if (x1Start <= x1Len && x2Start <= x2Len)
                    xN = x[x1Start:min(x1Len, x1End),
                            x2Start:min(x2Len, x2End), :]
                    ub = upper[x1Start:min(x1Len, x1End),
                            x2Start:min(x2Len, x2End), :]
                    lb = lower[x1Start:min(x1Len, x1End),
                            x2Start:min(x2Len, x2End), :]
                    (xN1Len, xN2Len, _) = size(xN)
                    weight = weights[1:xN1Len, 1:xN2Len, :, k]
                end
                tauList[i,j,k], kappaList[i,j,k], nonzeroIndicesList[i,j,k],
                    uNewList[i,j,k], lNewList[i,j,k] = neuronConv2dSign(m, xN, y[i,j,k],
                    weight, bias[k], ub, lb, image=image, preCut=preCut)
            end
        end
    end
    return y, tauList, kappaList, nonzeroIndicesList, uNewList, lNewList
end

# A MIP formulation for a single neuron.
function neuronConv2dSign(m::JuMP.Model, x::VarOrAff, yijk::VarOrAff,
                w::Array{T, 3}, b::U,
                upper::Array{V, 3}, lower::Array{W, 3}; image=true, preCut=true
                ) where{T<:Real, U<:Real, V<:Real, W<:Real}
    # initNN!(m)
    posIndices = findall(w .> 0)
    negIndices = findall(w .< 0)
    nonzeroIndices = union(posIndices, negIndices)
    nonzeroNum = length(nonzeroIndices)
    if (nonzeroNum == 0 || size(x) == (0,0,0))
        if (b >= 0)
            @constraint(m, yijk == 1)
        else
            @constraint(m, yijk == -1)
        end
        return b, b, nonzeroIndices, upper, lower
    end
    if (image)
        wSum = sum(w)
        tau, kappa = getTauAndKappa(nonzeroNum, 510*b + wSum)
        tau = (tau - wSum) / 510
        kappa = (kappa - wSum) / 510
    else
        tau, kappa= b, b
    end
    uNew, lNew = transformProc(negIndices, upper, lower)
    if (preCut && (nonzeroNum <= 5) )
        IsetAll = collect(powerset(nonzeroIndices))
        IsetAll = IsetAll[2:length(IsetAll)]
    else
        IsetAll = [union(posIndices, negIndices)]
    end
    for Iset in IsetAll
        IsetC = setdiff(nonzeroIndices, Iset)
        @constraint(m, 2 * (sum(w[id]*x[id] for id in Iset) +
                            sum(w[id] * uNew[id] for id in IsetC) + tau) >=
                            ((sum(w[id]*lNew[id] for id in Iset) +
                            sum(w[id] * uNew[id] for id in IsetC) + tau) * (1 - yijk)) )
        @constraint(m, 2 * sum(w[id]*(uNew[id] - x[id]) for id in Iset) >=
                            ((sum(w[id]*uNew[id] for id in Iset) +
                            sum(w[id]*lNew[id] for id in IsetC) + kappa) * (1 - yijk)) )
    end
    return tau, kappa, nonzeroIndices, uNew, lNew
end


# A transformation for Fourier-Motzkin procedure.
function transformProc(negIndices::Array{CartesianIndex{3}, 1},
            upper::Array{T, 3}, lower::Array{U, 3}) where {T<:Real, U<:Real}
    upperNew = deepcopy(upper)
    lowerNew = deepcopy(lower)
    for id in negIndices
        upperNew[id] = lower[id]
        lowerNew[id] = upper[id]
    end
    return upperNew, lowerNew
end
