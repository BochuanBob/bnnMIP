include("layers.jl")
using Gurobi
export getBNNoutput

# Add constraints such that y = NN(x).
# If cuts == false, it is a Big-M formulation.
# Otherwise, cuts for the ideal formulation are added to the model.
function getBNNoutput(m::JuMP.Model, nn, x::VarOrAff; cuts=true,
            image=true, preCut=true)
    global callbackTimeTotal = 0.0
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
        if (nn[i]["type"] == "flatten")
            xOut = flatten(m, xIn)
        elseif (nn[i]["type"] == "activation")
            xOut = activation1D(m, xIn, nn[i]["function"],
                    upper = nn[i]["upper"], lower = nn[i]["lower"])
        elseif (nn[i]["type"] == "denseBin")
            takeSign = false
            if (haskey(nn[i], "activation"))
                takeSign = (nn[i]["activation"] == "Sign")
            end
            # After the first sign() function, the input of each binary layer
            # has entries of -1 and 1.
            xOut, tauList, kappaList, oneIndicesList, negOneIndicesList =
                        denseBin(m, xIn, nn[i]["weights"], nn[i]["bias"],
                        takeSign=takeSign, image=image, preCut=preCut)
            nn[i]["tauList"] = tauList
            nn[i]["kappaList"] = kappaList
            nn[i]["oneIndicesList"] = oneIndicesList
            nn[i]["negOneIndicesList"] = negOneIndicesList
            nn[i]["takeSign"] = takeSign
            nn[i]["xIn"] = xIn
            nn[i]["xOut"] = xOut
        elseif (nn[i]["type"] == "dense")
            actFunc = ""
            if (haskey(nn[i], "activation"))
                actFunc = nn[i]["activation"]
            end
            xOut, tauList, kappaList, nonzeroIndicesList,
                        uNewList, lNewList =
                        dense(m, xIn, nn[i]["weights"], nn[i]["bias"],
                        nn[i]["upper"], nn[i]["lower"],
                        actFunc=actFunc, image=image, preCut=preCut)
            nn[i]["tauList"] = tauList
            nn[i]["kappaList"] = kappaList
            nn[i]["nonzeroIndicesList"] = nonzeroIndicesList
            nn[i]["uNewList"] = uNewList
            nn[i]["lNewList"] = lNewList
            nn[i]["takeSign"] = (actFunc=="Sign")
            nn[i]["xIn"] = xIn
            nn[i]["xOut"] = xOut
        elseif (nn[i]["type"] == "conv2dSign")
            strides = NTuple{2, Int64}(nn[i]["strides"])
            xOut, tauList, kappaList, nonzeroIndicesList,
                        uNewList, lNewList =
                        conv2dSign(m, xIn, nn[i]["weights"], nn[i]["bias"],
                        strides, nn[i]["upper"], nn[i]["lower"],
                        padding=nn[i]["padding"], image=image, preCut=preCut)
            nn[i]["tauList"] = tauList
            nn[i]["kappaList"] = kappaList
            nn[i]["nonzeroIndicesList"] = nonzeroIndicesList
            nn[i]["uNewList"] = uNewList
            nn[i]["lNewList"] = lNewList
            nn[i]["xIn"] = xIn
            nn[i]["xOut"] = xOut
        elseif (nn[i]["type"] == "conv2dBinSign")
            # After the first sign() function, the input of each binary layer
            # has entries of -1 and 1.
            strides = NTuple{2, Int64}(nn[i]["strides"])
            xOut, tauList, kappaList, oneIndicesList, negOneIndicesList =
                        conv2dBinSign(m, xIn, nn[i]["weights"], nn[i]["bias"],
                        strides, padding=nn[i]["padding"],
                        image=image, preCut=preCut)
            nn[i]["tauList"] = tauList
            nn[i]["kappaList"] = kappaList
            nn[i]["oneIndicesList"] = oneIndicesList
            nn[i]["negOneIndicesList"] = negOneIndicesList
            nn[i]["xIn"] = xIn
            nn[i]["xOut"] = xOut
        else
            error("Not support layer.")
        end
        xIn = xOut
    end
    y = xOut
    # Whether submit the cuts to Gurobi.
    if (cuts)
        global iter = 0
        # Generate cuts by callback function
        function callbackCutsBNN(cb_data)
            iter += 1
            if (iter>100 && mod(iter, 1000) != 1)
                return
            end
            callbackTime = @elapsed begin
                flag = true
                for i in 1:nnLen
                    # if (~flag)
                    #     break
                    # end
                    if (nn[i]["type"] == "denseBin" && nn[i]["takeSign"])
                        xIn = nn[i]["xIn"]
                        xOut = nn[i]["xOut"]
                        flag = addDenseBinCons!(m, xIn, xOut, nn[i]["tauList"],
                                            nn[i]["kappaList"],
                                            nn[i]["oneIndicesList"],
                                            nn[i]["negOneIndicesList"],
                                            cb_data)
                    elseif (nn[i]["type"] == "dense" && nn[i]["takeSign"])
                        xIn = nn[i]["xIn"]
                        xOut = nn[i]["xOut"]
                        flag = addDenseCons!(m, xIn, xOut, nn[i]["weights"],
                                        nn[i]["tauList"], nn[i]["kappaList"],
                                        nn[i]["nonzeroIndicesList"],
                                        nn[i]["uNewList"], nn[i]["lNewList"],
                                        cb_data, image=image)
                    elseif (nn[i]["type"] == "denseBinImage" && nn[i]["takeSign"])
                        xIn = nn[i]["xIn"]
                        xOut = nn[i]["xOut"]
                        flag = addDenseBinImageCons!(m, xIn, xOut, nn[i]["weights"],
                                                nn[i]["bias"], cb_data)
                    end
                end
            end
            callbackTimeTotal += callbackTime
        end
        MOI.set(m, MOI.UserCutCallback(), callbackCutsBNN)
    else
        function callback(cb_data)
            # callbackTime = @elapsed begin
            # end
            # callbackTimeTotal += callbackTime
            return
        end
        MOI.set(m, MOI.UserCutCallback(), callback)
    end
    return y
end
