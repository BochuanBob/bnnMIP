include("layers.jl")
include("callback.jl")
export getBNNoutput

const TESTCONST = Array{Array{Float64,2}, 1}([zeros(100, 100), ones(4,3), zeros(20, 100)])

# Add constraints such that y = NN(x).
# If cuts == false, it is a Big-M formulation.
# Otherwise, cuts for the ideal formulation are added to the model.
function getBNNoutput(m::JuMP.Model, nn::Array{NNLayer, 1}, x::VarOrAff; cuts=true,
            image=true, preCut=true)
    initNN!(m)
    count = m.ext[:NN].count
    m.ext[:NN].count += 1
    nnLen = length(nn)
    # Don't want to change the original data.
    nn = deepcopy(nn)
    xIn = x
    xOut = nothing
    xOnes = false
    for i in 1:nnLen
        if (typeof(nn[i]) == FlattenLayer)
            xOut = flatten(m, xIn)
            nn[i].xIn = xIn
            nn[i].xOut = xOut
        # elseif (nn[i]["type"] == "activation")
        #     xOut = activation1D(m, xIn, nn[i]["function"],
        #             upper = nn[i]["upper"], lower = nn[i]["lower"])
    elseif (typeof(nn[i]) == DenseBinLayer)
            takeSign = (nn[i].activation == "Sign")
            # After the first sign() function, the input of each binary layer
            # has entries of -1 and 1.
            xOut, tauList, kappaList, oneIndicesList, negOneIndicesList =
                        denseBin(m, xIn, nn[i].weights, nn[i].bias,
                        takeSign=takeSign, image=image, preCut=preCut,cuts=cuts)
            nn[i].tauList = tauList
            nn[i].kappaList = kappaList
            nn[i].oneIndicesList = oneIndicesList
            nn[i].negOneIndicesList = negOneIndicesList
            nn[i].takeSign = takeSign
            nn[i].xIn = xIn
            nn[i].xOut = xOut
        elseif (typeof(nn[i]) == DenseLayer)
            actFunc = nn[i].activation
            xOut, tauList, kappaList, nonzeroIndicesList,
                        uNewList, lNewList =
                        dense(m, xIn, nn[i].weights, nn[i].bias,
                        nn[i].upper, nn[i].lower,
                        actFunc=actFunc, image=image, preCut=preCut)
            nn[i].tauList = tauList
            nn[i].kappaList = kappaList
            nn[i].nonzeroIndicesList = nonzeroIndicesList
            nn[i].uNewList = uNewList
            nn[i].lNewList = lNewList
            nn[i].takeSign = (actFunc=="Sign")
            nn[i].xIn = xIn
            nn[i].xOut = xOut
        elseif (typeof(nn[i]) == Conv2dLayer)
            strides = nn[i].strides
            xOut, tauList, kappaList, nonzeroIndicesList,
                        uNewList, lNewList =
                        conv2dSign(m, xIn, nn[i].weights, nn[i].bias,
                        strides, nn[i].upper, nn[i].lower,
                        padding=nn[i].padding, image=image, preCut=preCut)
            nn[i].tauList = tauList
            nn[i].kappaList = kappaList
            nn[i].nonzeroIndicesList = nonzeroIndicesList
            nn[i].uNewList = uNewList
            nn[i].lNewList = lNewList
            nn[i].xIn = xIn
            nn[i].xOut = xOut
        elseif (typeof(nn[i]) == Conv2dBinLayer)
            # After the first sign() function, the input of each binary layer
            # has entries of -1 and 1.
            strides = nn[i].strides
            xOut, tauList, kappaList, oneIndicesList, negOneIndicesList =
                        conv2dBinSign(m, xIn, nn[i].weights, nn[i].bias,
                        strides, padding=nn[i].padding,
                        image=image, preCut=preCut,cuts=cuts)
            nn[i].tauList = tauList
            nn[i].kappaList = kappaList
            nn[i].oneIndicesList = oneIndicesList
            nn[i].negOneIndicesList = negOneIndicesList
            nn[i].xIn = xIn
            nn[i].xOut = xOut
        else
            error("Not support layer.")
        end
        xIn = xOut
    end
    y = xOut
    # Whether submit the cuts to Gurobi.
    if (cuts)
        # Generate cuts by callback function
        function callbackCutsBNN(cb_data, nnCopy::Array{NNLayer, 1})
            # callbackTime = @elapsed begin
                if (typeof(nnCopy[1]) == FlattenLayer)
                    xOut = nnCopy[1].xOut
                    weights = nnCopy[2].weights
                    xLen = length(xOut)
                    xInVal = aff_callback_value.(Ref(cb_data), xOut)
                elseif (typeof(nnCopy[1]) == Conv2dLayer)
                    xIn = nnCopy[1].xIn
                    xOut = nnCopy[1].xOut
                    strides = nnCopy[1].strides
                    xInVal, flag = addConv2dCons!(m, xIn, xOut, nnCopy[1].weights,
                                    nnCopy[1].tauList, nnCopy[1].kappaList,
                                    nnCopy[1].nonzeroIndicesList,
                                    nnCopy[1].uNewList, nnCopy[1].lNewList,
                                    strides, cb_data, image=image)
                end

                for i in 2:nnLen
                    if (typeof(nn[i]) == DenseBinLayer && nn[i].takeSign)
                        xIn = nn[i].xIn
                        xOut = nn[i].xOut
                        xInVal, flag = addDenseBinCons!(m, xIn, xInVal,
                                            xOut, nn[i].tauList,
                                            nn[i].kappaList,
                                            nn[i].oneIndicesList,
                                            nn[i].negOneIndicesList,
                                            cb_data)
                    elseif (typeof(nn[i]) == Conv2dBinLayer)
                        xIn = nn[i].xIn
                        xOut = nn[i].xOut
                        strides = nn[i].strides
                        xInVal, flag = addConv2dBinCons!(m, xIn, xInVal, xOut,
                                            nn[i].tauList,
                                            nn[i].kappaList,
                                            nn[i].oneIndicesList,
                                            nn[i].negOneIndicesList,
                                            nn[i].weights, strides,
                                            cb_data)
                    elseif (typeof(nn[i]) == DenseLayer && nn[i].takeSign)
                        xIn = nn[i].xIn
                        xOut = nn[i].xOut
                        xInVal, flag = addDenseCons!(m, xIn, xInVal, xOut,
                                        nn[i].weights,
                                        nn[i].tauList, nn[i].kappaList,
                                        nn[i].nonzeroIndicesList,
                                        nn[i].uNewList, nn[i].lNewList,
                                        cb_data, image=image)
                    elseif (typeof(nn[i]) == Flatten)
                        xOut = nn[i].xOut
                        xLen = length(xOut)
                        xInVal = zeros(xLen)
                        for j in 1:xLen
                            xInVal = aff_callback_value(cb_data, xOut[j])
                        end
                    end
                end
            end
            m.ext[:CALLBACK_TIME].time += callbackTime
        end
        set(backend(m), MOI.UserCutCallback(), callbackCutsBNN)
    else
        function callback(cb_data, nn::Array{NNLayer, 1})
            return
        end
        set(backend(m), MOI.UserCutCallback(), callback)
    end
    return y, nn
end
