include("layers.jl")
include("forwardProp.jl")
include("userCuts.jl")
export getBNNoutput

# Add constraints such that y = NN(x).
# If cuts == false, it is a Big-M formulation.
# Otherwise, cuts for the ideal formulation are added to the model.
function getBNNoutput(m::JuMP.Model, nn::Array{NNLayer, 1},
            x::Array{JuMP.VariableRef}; cuts=true,
            image=true, preCut=true)
    initNN!(m)
    count = m.ext[:NN].count
    m.ext[:NN].count += 1
    nnLen = length(nn)
    # Don't want to change the original data.
    # nn = deepcopy(nn)
    xIn = x
    xOut = Array{JuMP.VariableRef, 1}([])
    xOnes = false
    upper= Array{Float64, 1}([])
    lower = Array{Float64, 1}([])
    zList = Array{JuMP.VariableRef, 1}([])
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
            xOut, z, tauList, kappaList, oneIndicesList, negOneIndicesList =
                        denseBin(m, xIn, nn[i].weights, nn[i].bias,
                        takeSign=takeSign, image=image, preCut=preCut,cuts=cuts, layer=nnLen - i)
            nn[i].tauList = tauList
            nn[i].kappaList = kappaList
            nn[i].oneIndicesList = oneIndicesList
            nn[i].negOneIndicesList = negOneIndicesList
            nn[i].takeSign = takeSign
            nn[i].xIn = xIn
            nn[i].xOut = xOut
            if (takeSign)
                append!(zList, z)
            end
        elseif (typeof(nn[i]) == DenseLayer)
            actFunc = nn[i].activation
            xOut, z, tauList, kappaList, nonzeroIndicesList,
                        uNewList, lNewList =
                        dense(m, xIn, nn[i].weights, nn[i].bias,
                        nn[i].upper, nn[i].lower,
                        actFunc=actFunc, image=image, preCut=preCut, layer=nnLen - i)
            nn[i].tauList = tauList
            nn[i].kappaList = kappaList
            nn[i].nonzeroIndicesList = nonzeroIndicesList
            nn[i].uNewList = uNewList
            nn[i].lNewList = lNewList
            nn[i].takeSign = (actFunc=="Sign")
            println("Dense: ", nn[i].takeSign)
            nn[i].xIn = xIn
            nn[i].xOut = xOut
            upper = nn[i].upper
            lower = nn[i].lower
            if (nn[i].takeSign)
                append!(zList, z)
            end
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
            upper = nn[i].upper
            lower = nn[i].lower
            append!(zList, z[:])
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
            append!(zList, z[:])
        else
            error("Not support layer.")
        end
        xIn = xOut
    end
    y = xOut
    # Whether submit the cuts to Gurobi.
    if (cuts)
        # Generate cuts by callback function
        function callbackCutsBNN(cb_data)
            callbackFunc(m, cb_data, nn)
        end
        MOI.set(m, MOI.UserCutCallback(), callbackCutsBNN)
    else
        function callback(cb_data)
            return
        end
        MOI.set(m, MOI.UserCutCallback(), callback)
    end
    iter = 1
    function heuristicBNN(cb_data)
        input = aff_callback_value.(Ref(cb_data), xInList[1])
        outputVal = forwardPropBNN(input, nn, length(zList))
        append!(zList, flatten(xInList[1]))
        append!(zList, xOutList[nnLen])
        time = @elapsed begin
        status = MOI.submit(
            m, MOI.HeuristicSolution(cb_data),
                zList,
                outputVal
        )
        end
        m.ext[:TEST_CONSTRAINTS].count += 1
        m.ext[:BENCH_CONV2D].time += time
        if (status == MOI.HEURISTIC_SOLUTION_ACCEPTED)
            println("I submitted a heuristic solution, and the status was: ", status)
        end
    end
    # MOI.set(m, MOI.HeuristicCallback(), heuristicBNN)
    return y, nn
end
