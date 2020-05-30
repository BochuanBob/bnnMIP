using JuMP, Gurobi
include("../src/neuralNetworks/layers/denseBin.jl")
include("testFunc.jl")

function testDenseBin!(xInput, weights::Array{T, 2}, bias::Array{U, 1};
               takeSign=false) where{T<:Real, U<:Real}
    m = direct_model(Gurobi.Optimizer(OutputFlag=0, Method=1))
    (yLen, xLen) = size(weights)
    x = @variable(m, [1:xLen], base_name="x")
    @constraint(m, [i=1:xLen], x[i] == xInput[i])
    y, _, _, _, _ = denseBin(m, x, weights, bias, takeSign=takeSign)
    @objective(m, Min, 0)
    optimize!(m)
    if(takeSign)
        println("The expected output value: ", signFun.(weights * xInput
                                + bias))
    else
        println("The expected output value: ", weights * xInput
                                + bias)
    end
    println("The actual output value: ", value.(y))
    return nothing
end

weights = Array{Float64, 2}([[-1 1 -1]; [-1 1 1]])
bias = Array{Float64, 1}([0, 0])
xInput = Array{Float64, 1}([1, 1, 1])
testDenseBin!(xInput, weights, bias, takeSign=true)
testDenseBin!(xInput, weights, bias, takeSign=false)

bias = Array{Float64, 1}([0.1, -3.4])
testDenseBin!(xInput, weights, bias, takeSign=true)
testDenseBin!(xInput, weights, bias, takeSign=false)

bias = Array{Float64, 1}([1, -5])
testDenseBin!(xInput, weights, bias, takeSign=true)
testDenseBin!(xInput, weights, bias, takeSign=false)

bias = Array{Float64, 1}([3, -4])
testDenseBin!(xInput, weights, bias, takeSign=true)
testDenseBin!(xInput, weights, bias, takeSign=false)
