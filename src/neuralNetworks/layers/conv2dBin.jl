include("layerSetup.jl")
include("denseSetup.jl")
export conv2dBinSign

# A conv2d layer with sign().
# Each entry of weights must be -1, 0, 1.
function conv2dBinSign(m::JuMP.Model, x::VarOrAff,
                weights::Array{T, 4}, bias::Array{U, 1},
                strides::Tuple{Int64, Int64}; padding="valid",
                image=true, preCut=true) where{T<:Real, U<:Real}
    if (~checkWeights(weights))
        error("Each entry of weights must be -1, 0, 1.")
    end
    EMPTY3DVAR = Array{VariableRef, 3}(undef, (0,0,0))
    EMPTY3DFLOAT = Array{Float64, 3}(undef, (0,0,0))
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

    outputSize = (y1Len, y2Len, y3Len)
    tauList = zeros(outputSize)
    kappaList = zeros(outputSize)
    oneIndicesList = Array{Array{CartesianIndex{3}, 1}, 3}(undef, outputSize)
    negOneIndicesList = Array{Array{CartesianIndex{3}, 1}, 3}(undef, outputSize)
    z = @variable(m, [1:y1Len, 1:y2Len, 1:y3Len], binary=true,
                base_name="z_$count")
    y = @expression(m, 2 .* z .- 1)
    for i in 1:y1Len
        for j in 1:y2Len
            for k in 1:y3Len
                weight = EMPTY3DFLOAT
                x1Start, x1End = 1 + (i-1)*s1Len, (i-1)*s1Len + k1Len
                x2Start, x2End = 1 + (j-1)*s2Len, (j-1)*s2Len + k2Len
                xN = EMPTY3DVAR
                if (x1Start <= x1Len && x2Start <= x2Len)
                    xN = x[x1Start:min(x1Len, x1End),
                            x2Start:min(x2Len, x2End), :]
                    (xN1Len, xN2Len, _) = size(xN)
                    weight = weights[1:xN1Len, 1:xN2Len, :, k]
                end
                tauList[i,j,k], kappaList[i,j,k], oneIndicesList[i,j,k],
                    negOneIndicesList[i,j,k] = neuronSign(m, xN, y[i,j,k],
                            weight, bias[k], image=image, preCut=preCut)
            end
        end
    end
    return y, tauList, kappaList, oneIndicesList, negOneIndicesList
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
function neuronSign(m::JuMP.Model, x::VarOrAff, yijk::VarOrAff,
                weightVec::Array{T, 3}, b::U;
                image=true, preCut=true) where{T<:Real, U<:Real}
    # initNN!(m)
    oneIndices = findall(weightVec .== 1)
    negOneIndices = findall(weightVec .== -1)
    nonzeroNum = length(oneIndices) + length(negOneIndices)

    if (nonzeroNum == 0 || size(x) == (0,0,0))
        if (b >= 0)
            @constraint(m, yi == 1)
        else
            @constraint(m, yi == -1)
        end
        return b, b, oneIndices, negOneIndices
    end
    # For comparison to not handling discontinuity.
    if (image)
        tau, kappa = getTauAndKappa(nonzeroNum, b)
    else
        tau, kappa = b, b
    end
    if (tau >= nonzeroNum)
        @constraint(m, yi == 1)
        return tau, kappa, oneIndices, negOneIndices
    end
    if (kappa <= -nonzeroNum)
        @constraint(m, yi == -1)
        return tau, kappa, oneIndices, negOneIndices
    end
    if (preCut && (nonzeroNum <= 6) )
        IsetAll = collect(powerset(union(oneIndices, negOneIndices)))
        IsetAll = IsetAll[2:length(IsetAll)]
    else
        IsetAll = [union(oneIndices, negOneIndices)]
    end
    for Iset1 in IsetAll
        Ipos = intersect(Iset1, oneIndices)
        Ineg = intersect(Iset1, negOneIndices)
        lenI = length(Iset1)
        @constraint(m, (sum(x[id] for id in Ipos) - sum(x[id] for id in Ineg))
                        >=(((lenI - nonzeroNum - tau) * (1 + yijk) /2)
                        - (1 - yijk) * lenI/2 ) )
        @constraint(m, (sum(x[id] for id in Ipos) - sum(x[id] for id in Ineg))
                    <=(((1 + yijk) * lenI/2)
                    - (lenI - nonzeroNum + kappa)*(1 - yijk)/2) )
    end
    return tau, kappa, oneIndices, negOneIndices
end
