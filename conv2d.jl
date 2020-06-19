include("layerSetup.jl")
include("activation.jl")
include("denseSetup.jl")
export conv2dSign

const CUTOFF_CONV2D = 16
const CUTOFF_CONV2D_PRECUT = 5

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
    EMPTY3DVAR = Array{VariableRef, 3}(undef, (0,0,0))
    EMPTY3DFLOAT = Array{Float64, 3}(undef, (0,0,0))
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
    tauList = zeros(y3Len)
    kappaList = zeros(y3Len)
    nonzeroIndicesList = Array{Array{CartesianIndex{3}, 1}, 3}(undef, (k1Len, k2Len, y3Len))
    for i in 1:k1Len
        for j in 1:k2Len
            for k in 1:y3Len
                nonzeroIndicesList[i,j,k] = findall(weights[1:i, 1:j, :, k] .!= 0)
            end
        end
    end
    uNewList = Array{Array{Float64, 3}, 3}(undef, outputSize)
    lNewList = Array{Array{Float64, 3}, 3}(undef, outputSize)
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
                tauList[k], kappaList[k],
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
    if (image)
        wSum = sum(w)
        tau, kappa = getTauAndKappa(nonzeroNum, 510*b + wSum)
        tau = (tau - wSum) / 510
        kappa = (kappa - wSum) / 510
    else
        tau, kappa= b, b
    end
    if (nonzeroNum == 0 || size(x) == (0,0,0))
        if (b >= 0)
            @constraint(m, yijk == 1)
        else
            @constraint(m, yijk == -1)
        end
        return tau, kappa, upper, lower
    end
    uNew, lNew = transformProc(negIndices, upper, lower)
    if (preCut && (nonzeroNum <= CUTOFF_CONV2D_PRECUT) )
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
    return tau, kappa, uNew, lNew
end


# A transformation on bounds for Fourier-Motzkin procedure.
function transformProc(negIndices::Array{CartesianIndex{3}, 1},
            upper::Array{T, 3}, lower::Array{U, 3}) where {T<:Real, U<:Real}
    upperNew = deepcopy(upper)
    lowerNew = deepcopy(lower)
    for id in negIndices
        upperNew[id] = lower[id]
        lowerNew[id] = upper[id]
    end
    return Float64.(upperNew), Float64.(lowerNew)
end

function addConv2dCons!(m::JuMP.Model, xIn::VarOrAff, xOut::VarOrAff,
                        weights::Array{Float64, 4},tauList::Array{Float64, 1},
                        kappaList::Array{Float64, 1},
                        nonzeroIndicesList::Array{Array{CartesianIndex{3}, 1}, 3},
                        uNewList::Array{Array{Float64, 3}, 3},
                        lNewList::Array{Array{Float64, 3}, 3},
                        strides::Tuple{Int64, Int64},
                        cb_data; image=true)
    (x1Len, x2Len, x3Len) = size(xIn)
    (y1Len, y2Len, y3Len) = size(xOut)
    (k1Len, k2Len, channels, filters) = size(weights)
    (s1Len, s2Len) = strides
    xVal = zeros((x1Len, x2Len, x3Len))
    for i in 1:x1Len
        for j in 1:x2Len
            for k in 1:x3Len
                xVal[i,j,k] = JuMP.callback_value(cb_data, xIn[i,j,k])
            end
        end
    end
    contFlag = true
    yVal = zeros((y1Len, y2Len, y3Len))
    K = 2
    iter = 0
    for i in 1:y1Len
        for j in 1:y2Len
            for k in 1:y3Len
                # if (iter > K)
                #     return contFlag
                # end
                # idy = CartesianIndex{3}(i,j,k)
                yVal[i,j,k] = aff_callback_value(cb_data, xOut[i,j,k])
                if (-0.99 >= yVal[i,j,k] || yVal[i,j,k] >= 0.99)
                    continue
                end
                x1Start, x1End = 1 + (i-1)*s1Len, (i-1)*s1Len + k1Len
                x2Start, x2End = 1 + (j-1)*s2Len, (j-1)*s2Len + k2Len
                x1NEnd = min(x1Len, x1End)
                x2NEnd = min(x2Len, x2End)
                nonzeroIndices = nonzeroIndicesList[(x1NEnd-x1Start+1),(x2NEnd-x2Start+1),k]
                nonzeroNum = length(nonzeroIndices)
                if (nonzeroNum == 0)
                    continue
                end
                wVec = weights[:, :, :, k]
                tau, kappa = tauList[k], kappaList[k]
                uNew, lNew = uNewList[i,j,k], lNewList[i,j,k]
                xValN = xVal[x1Start:x1NEnd,
                        x2Start:x2NEnd, :]
                xInN = xIn[x1Start:x1NEnd,
                        x2Start:x2NEnd, :]
                con1Val, con2Val = decideViolationCons(xValN, yVal[i,j,k],nonzeroIndices,
                                    wVec, tau, kappa, uNew,lNew)

                if (con1Val > 0)
                    I1 = getFirstCutIndices(xValN, yVal[i,j,k],nonzeroIndices,wVec,
                                            uNew,lNew)
                    if (length(I1) <= CUTOFF_CONV2D)
                        con1 = getFirstCon(xInN, xOut[i,j,k], I1,
                                    nonzeroIndices, wVec, tau, uNew, lNew)
                        MOI.submit(m, MOI.UserCut(cb_data), con1)
                        m.ext[:CUTS].count += 1
                        iter += 1
                    end
                end
                if (con2Val > 0)
                    I2 = getSecondCutIndices(xValN, yVal[i,j,k],nonzeroIndices,wVec,
                                            uNew,lNew)
                    if (length(I2) <= CUTOFF_CONV2D)
                        con2 = getSecondCon(xInN, xOut[i,j,k], I2,
                                    nonzeroIndices, wVec, kappa, uNew, lNew)
                        MOI.submit(m, MOI.UserCut(cb_data), con2)
                        m.ext[:BENCH_CONV2D].time += time
                        m.ext[:CUTS].count += 1
                        iter += 1
                    end
                end
            end
        end
    end
    return contFlag
end

function decideViolationCons(xVal::Array{Float64, 3}, yVal::Float64,
                        nonzeroIndices::Array{CartesianIndex{3},1},
                        w::Array{Float64, 3}, tau::Float64, kappa::Float64,
                        upper::Array{Float64, 3}, lower::Array{Float64, 3}
                        )
    # Initially, I = []
    con1Val = 2 * (sum(w[i] * upper[i] for i in nonzeroIndices) + tau) -
                (sum(w[i] * upper[i] for i in nonzeroIndices) + tau) * (1 - yVal)
    con2Val = - (sum(w[i] * lower[i] for i in nonzeroIndices) + kappa) * (1 - yVal)
    for i in nonzeroIndices
        con1Delta = 2*w[i]*(xVal[i] - upper[i]) - w[i]*(lower[i]-upper[i])*(1-yVal)
        con2Delta = 2*w[i]*(upper[i]-xVal[i]) - w[i]*(upper[i]-lower[i])*(1-yVal)
        con1Val = con1Val + min(0, con1Delta)
        con2Val = con2Val + min(0, con2Delta)
    end
    return -con1Val, -con2Val
end

# Output I^1 for the first constraint in Proposition 1
function getFirstCutIndices(xVal::Array{Float64, 3}, yVal::Float64,
                        nonzeroIndices::Array{CartesianIndex{3},1},
                        w::Array{Float64, 3},
                        upper::Array{Float64, 3}, lower::Array{Float64, 3}
                        )
    I1 = Array{CartesianIndex{3},1}([])
    for i in nonzeroIndices
        if (2*w[i]*(xVal[i] - upper[i]) < w[i]*(lower[i]-upper[i])*(1-yVal))
            I1 = vcat(I1, i)
        end
    end
    return I1
end

# Output I^2 for the second constraint in Proposition 1
function getSecondCutIndices(xVal::Array{Float64, 3}, yVal::U,
                        nonzeroIndices::Array{CartesianIndex{3},1},
                        w::Array{Float64, 3},
                        upper::Array{Float64, 3}, lower::Array{Float64, 3}
                        )
    I2 = Array{CartesianIndex{3},1}([])
    for i in nonzeroIndices
        if (2*w[i]*(upper[i]-xVal[i]) < w[i]*(upper[i]-lower[i])*(1-yVal))
            I2 = vcat(I2, i)
        end
    end
    return I2
end

# Return first constraint with given I, I^+, I^-, tau, kappa as shown
# in Proposition 1. An efficient implementation for user cuts.
function getFirstCon(x::VarOrAff, yijk::VarOrAff,
                            Iset::Array{CartesianIndex{3},1},
                            nonzeroIndices::Array{CartesianIndex{3},1},
                            w::Array{Float64, 3},b::Float64,
                            upper::Array{Float64, 3}, lower::Array{Float64, 3}
                            )
    IsetC = setdiff(nonzeroIndices, Iset)
    return @build_constraint(2 * (sum(w[i]*x[i] for i in Iset) +
                        sum(w[i] * upper[i] for i in IsetC) + b) >=
                        (sum(w[i]*lower[i] for i in Iset) +
                        sum(w[i] * upper[i] for i in IsetC) + b) * (1 - yijk)
                        )
end

# Return second constraint with given I, I^+, I^-, tau, kappa as shown
# in Proposition 1. An efficient implementation for user cuts.
function getSecondCon(x::VarOrAff, yijk::VarOrAff,
                            Iset::Array{CartesianIndex{3},1},
                            nonzeroIndices::Array{CartesianIndex{3},1},
                            w::Array{Float64, 3},b::Float64,
                            upper::Array{Float64, 3}, lower::Array{Float64, 3}
                            )
    IsetC = setdiff(nonzeroIndices, Iset)
    return @build_constraint(2 * sum(w[i]*(upper[i] - x[i]) for i in Iset) >=
                        (sum(w[i]*upper[i] for i in Iset) +
                        sum(w[i]*lower[i] for i in IsetC) + b) * (1 - yijk)
                        )
end
