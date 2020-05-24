include("layers.jl")
export getBNNoutput

# Add constraints such that y = NN(x).
# If cuts == false, it is a Big-M formulation.
# Otherwise, cuts for the ideal formulation are added to the model.
function getBNNoutput(m::JuMP.Model, nn, x::VarOrAff; cuts=true, image=true)
    global callbackTimeTotal = 0.0
    initNN!(m)
    count = m.ext[:NN].count
    m.ext[:NN].count += 1
    nnLen = length(nn)
    # Don't want to change the original data.
    nn = copy(nn)
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
            xOut = denseBin(m, xIn, nn[i]["weights"], nn[i]["bias"],
                        takeSign=takeSign, image=image)
            nn[i]["takeSign"] = takeSign
            nn[i]["xIn"] = xIn
            nn[i]["xOut"] = xOut
        elseif (nn[i]["type"] == "dense")
            actFunc = ""
            if (haskey(nn[i], "activation"))
                actFunc = nn[i]["activation"]
            end
            xOut = dense(m, xIn, nn[i]["weights"], nn[i]["bias"],
                        nn[i]["upper"], nn[i]["lower"],
                        actFunc=actFunc, image=image)
            nn[i]["takeSign"] = (actFunc=="Sign")
            nn[i]["xIn"] = xIn
            nn[i]["xOut"] = xOut
        elseif (nn[i]["type"] == "denseBinImage")
            actFunc = ""
            if (haskey(nn[i], "activation"))
                actFunc = nn[i]["activation"]
            end
            xxIn, xOut = denseBinImage(m, xIn, nn[i]["weights"], nn[i]["bias"],
                        takeSign=(actFunc=="Sign"))
            nn[i]["takeSign"] = (actFunc=="Sign")
            nn[i]["xIn"] = xxIn
            nn[i]["xOut"] = xOut
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
            callbackTime = @elapsed begin
                for i in 1:nnLen
                    if (nn[i]["type"] == "denseBin" && nn[i]["takeSign"])
                        xIn = nn[i]["xIn"]
                        xOut = nn[i]["xOut"]
                        addDenseBinCons!(m, xIn, xOut, nn[i]["weights"],
                                            nn[i]["bias"], cb_data, image=image)
                    elseif (nn[i]["type"] == "dense" && nn[i]["takeSign"])
                        xIn = nn[i]["xIn"]
                        xOut = nn[i]["xOut"]
                        addDenseCons!(m, xIn, xOut, nn[i]["weights"],
                                            nn[i]["bias"], nn[i]["upper"],
                                            nn[i]["lower"], cb_data, image=image)
                    elseif (nn[i]["type"] == "denseBinImage" && nn[i]["takeSign"])
                        xIn = nn[i]["xIn"]
                        xOut = nn[i]["xOut"]
                        addDenseBinImageCons!(m, xIn, xOut, nn[i]["weights"],
                                                nn[i]["bias"], cb_data)
                    end
                end
            end
            callbackTimeTotal += callbackTime
        end
        MOI.set(m, MOI.UserCutCallback(), callbackCutsBNN)
    end
    return y
end
