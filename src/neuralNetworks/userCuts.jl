export callbackFunc

function processLayer(m::JuMP.Model, layer::FlattenLayer, cb_data,
            xInVal::Array{Float64})
    xOut = layer.xOut
    time = @elapsed begin
        xInVal = JuMP.callback_value.(Ref(cb_data), xOut)
    end
    m.ext[:BENCH_CONV2D].time += time
    return xInVal
end

function processLayer(m::JuMP.Model, layer::Conv2dLayer, cb_data,
            xInVal::Array{Float64})
    xIn = layer.xIn
    xOut = layer.xOut
    strides = layer.strides
    xInVal, flag = addConv2dCons!(m, xIn, xOut, layer.weights,
                    layer.tauList, layer.kappaList,
                    layer.nonzeroIndicesList,
                    layer.uNewList, layer.lNewList,
                    strides, cb_data, image=image)
    return xInVal
end

function processLayer(m::JuMP.Model, layer::Conv2dBinLayer, cb_data,
            xInVal::Array{Float64})
    xIn = layer.xIn
    xOut = layer.xOut
    strides = layer.strides
    xInVal, flag = addConv2dBinCons!(m, xIn, xInVal, xOut,
                        layer.tauList,
                        layer.kappaList,
                        layer.oneIndicesList,
                        layer.negOneIndicesList,
                        layer.weights, strides,
                        cb_data)
    return xInVal
end

function processLayer(m::JuMP.Model, layer::DenseLayer, cb_data,
            xInVal::Array{Float64, 1})
    if (~layer.takeSign)
        return xInVal
    end
    xIn = layer.xIn
    xOut = layer.xOut
    xInVal, flag = addDenseCons!(m, xIn, xInVal, xOut,
                    layer.weights,
                    layer.tauList, layer.kappaList,
                    layer.nonzeroIndicesList,
                    layer.uNewList, layer.lNewList,
                    cb_data)
    return xInVal
end

function processLayer(m::JuMP.Model, layer::DenseBinLayer, cb_data,
            xInVal::Array{Float64, 1})
    if (~layer.takeSign)
        return xInVal
    end
    xIn = layer.xIn
    xOut = layer.xOut
    xInVal, flag = addDenseBinCons!(m, xIn, xInVal,
                        xOut, layer.tauList,
                        layer.kappaList,
                        layer.oneIndicesList,
                        layer.negOneIndicesList,
                        cb_data)
    return xInVal
end


function callbackFunc(m::JuMP.Model, cb_data, nn::Array{NNLayer, 1})
    callbackTime = @elapsed begin
    nnLen = length(nn)
    xInVal = Array{Float64, 1}([])
        for i in 1:nnLen
            xInVal = processLayer(m, nn[i], cb_data, xInVal)
        end
    end
    m.ext[:CALLBACK_TIME].time += callbackTime
end
