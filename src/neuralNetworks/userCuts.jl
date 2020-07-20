include("callbacks.jl")

export callbackFunc

const XIN_EMPTY = Array{Float64, 3}(undef, (0,0,0))

function processLayer(m::JuMP.Model, opt::Gurobi.Optimizer,
            layer::FlattenLayer, cb_data::Gurobi.CallbackData,
            xInVal::Array{Float64, 3},
            useDense::Bool, consistDense::Bool, consistDenseBin::Bool,
            Kval::Int64)
    xOut = layer.xOut
    if (xInVal !== XIN_EMPTY)
        return xInVal[:]::Array{Float64, 1}
    end
    xOutLen = length(xOut)
    xOutVal = zeros(xOutLen)
    # time = @elapsed begin
        for i in 1:xOutLen
            xOutVal[i] = my_callback_value(opt, cb_data, xOut[i])
        end
    # end
    # m.ext[:BENCH_CONV2D].time += time
    return xOutVal::Array{Float64, 1}
end

function processLayer(m::JuMP.Model, opt::Gurobi.Optimizer,
            layer::Conv2dLayer, cb_data::Gurobi.CallbackData,
            xInVal::Array{Float64, 3},
            useDense::Bool, consistDense::Bool, consistDenseBin::Bool,
            Kval::Int64)
    xIn = layer.xIn
    xOut = layer.xOut
    strides = layer.strides
    xInVal, flag = addConv2dCons!(m, opt, xIn, xOut, layer.weights,
                    layer.tauList, layer.kappaList,
                    layer.nonzeroIndicesList,
                    layer.uNewList, layer.lNewList,
                    strides, cb_data)
    return xInVal
end

function processLayer(m::JuMP.Model, opt::Gurobi.Optimizer,
            layer::Conv2dBinLayer, cb_data::Gurobi.CallbackData,
            xInVal::Array{Float64, 3},
            useDense::Bool, consistDense::Bool, consistDenseBin::Bool,
            Kval::Int64)
    xIn = layer.xIn
    xOut = layer.xOut
    strides = layer.strides
    xInVal, flag = addConv2dBinCons!(m, opt, xIn, xInVal, xOut,
                        layer.tauList,
                        layer.kappaList,
                        layer.oneIndicesList,
                        layer.negOneIndicesList,
                        layer.weights, strides,
                        cb_data)
    return xInVal
end

function processLayer(m::JuMP.Model, opt::Gurobi.Optimizer,
            layer::DenseLayer, cb_data::Gurobi.CallbackData,
            xInVal::Array{Float64, 1},
            useDense::Bool, consistDense::Bool, consistDenseBin::Bool,
            Kval::Int64)
    if (~layer.takeSign)
        return xInVal
    end
    xIn = layer.xIn
    xOut = layer.xOut
    xInVal, flag = addDenseCons!(m, opt, xIn, xInVal, xOut,
                    layer.weights,
                    layer.tauList, layer.kappaList,
                    layer.nonzeroIndicesList,
                    layer.uNewList, layer.lNewList,
                    cb_data, useDense, consistDense, Kval)
    return xInVal
end

function processLayer(m::JuMP.Model, opt::Gurobi.Optimizer,
            layer::DenseBinLayer, cb_data::Gurobi.CallbackData,
            xInVal::Array{Float64, 1},
            useDense::Bool, consistDense::Bool, consistDenseBin::Bool,
            Kval::Int64)
    if (~layer.takeSign)
        return xInVal
    end
    xIn = layer.xIn
    xOut = layer.xOut
    xInVal, flag = addDenseBinCons!(m, opt, xIn, xInVal,
                        xOut, layer.tauList,
                        layer.kappaList,
                        layer.oneIndicesList,
                        layer.negOneIndicesList,
                        cb_data, consistDenseBin, Kval)
    return xInVal
end


function callbackFunc(m::JuMP.Model, opt::Gurobi.Optimizer,
                cb_data::Gurobi.CallbackData, nn::Array{NNLayer, 1},
                useDense::Bool, consistDense::Bool, consistDenseBin::Bool,
                Kval::Int64)
    callbackTime = @elapsed begin
    nnLen = length(nn)
    xInVal = XIN_EMPTY
        for i in 1:nnLen
            xInVal = processLayer(m, opt, nn[i], cb_data, xInVal, useDense,
                                    consistDense, consistDenseBin, Kval)
        end
    end
    m.ext[:CALLBACK_TIME].time += callbackTime
end
