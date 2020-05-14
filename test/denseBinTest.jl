using JuMP, Gurobi
include("../src/neuralNetworks/layers/denseBin.jl")

function testDenseBin!(xInput, weights::Array{T, 2}, bias::Array{U, 1};
               takeSign=false, cuts=true, xOnes=true) where{T<:Real, U<:Real}
    m = direct_model(Gurobi.Optimizer(OutputFlag=0))
    (yLen, xLen) = size(weights)
    x = @variable(m, [1:xLen], base_name="x")
    @constraint(m, [i=1:xLen], x[i] == xInput[i])
    y = denseBin(m, x, weights, bias, takeSign=takeSign, cuts=cuts, xOnes=xOnes)
    @objective(m, Min, 0)
    optimize!(m)
    if(takeSign)
        println("The expected output value: ", sign.(weights * xInput
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
testDenseBin!(xInput, weights, bias, takeSign=true, cuts=false)
testDenseBin!(xInput, weights, bias, takeSign=true, cuts=true)
testDenseBin!(xInput, weights, bias, takeSign=false)

bias = Array{Float64, 1}([0.1, -3.4])
testDenseBin!(xInput, weights, bias, takeSign=true, cuts=false)
testDenseBin!(xInput, weights, bias, takeSign=true, cuts=true)
testDenseBin!(xInput, weights, bias, takeSign=false)

bias = Array{Float64, 1}([1, -5])
testDenseBin!(xInput, weights, bias, takeSign=true, cuts=false)
testDenseBin!(xInput, weights, bias, takeSign=true, cuts=true)
testDenseBin!(xInput, weights, bias, takeSign=false)

bias = Array{Float64, 1}([3, -4])
testDenseBin!(xInput, weights, bias, takeSign=true, cuts=false)
testDenseBin!(xInput, weights, bias, takeSign=true, cuts=true)
testDenseBin!(xInput, weights, bias, takeSign=false)


weights = Array{Float64, 2}([[-1 1 -1 -1]; [-1 1 -1 -1]])
bias = Array{Float64, 1}([0, 0])
xInput = Array{Float64, 1}([2, 10, -1, 1])
testDenseBin!(xInput, weights, bias, takeSign=false, xOnes=false)
testDenseBin!(xInput, weights, bias, takeSign=true, cuts=false, xOnes=false)
testDenseBin!(xInput, weights, bias, takeSign=true, cuts=true, xOnes=false)
