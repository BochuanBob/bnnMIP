using JuMP, Gurobi
include("../src/neuralNetworks/layers/dense.jl")
include("testFunc.jl")

function testDense!(xInput, weights::Array{T, 2}, bias::Array{U, 1},
               upper::Array{V, 1}, lower::Array{W, 1};
               actFunc="Sign") where{T<:Real, U<:Real, V<:Real, W<:Real}
    m = direct_model(Gurobi.Optimizer(OutputFlag=0))
    (yLen, xLen) = size(weights)
    x = @variable(m, [1:xLen], base_name="x")
    @constraint(m, [i=1:xLen], x[i] == xInput[i])
    y = dense(m, x, weights, bias, upper, lower, actFunc=actFunc)
    @objective(m, Min, 0)
    optimize!(m)
    if(actFunc=="Sign")
        println("The expected output value: ", signFun.(weights * xInput
                                + bias))
    else
        println("The expected output value: ", weights * xInput
                                + bias)
    end
    println("The actual output value: ", value.(y))
    return nothing
end

weights = Array{Float64, 2}([[1 2 3]; [30 1 10]])
bias = Array{Float64, 1}([0, -100])
xInput = Array{Float64, 1}([-3, 2, -4])
upper = Array{Float64, 1}([10, 10, 10])
lower = Array{Float64, 1}([-10, -10, -10])
testDense!(xInput, weights, bias, upper, lower)
testDense!(xInput, weights, bias,upper, lower, actFunc="")

weights = Array{Float64, 2}([[-1 2 -3]; [30 -1 10]])
xInput = Array{Float64, 1}([3, -2, -4])
testDense!(xInput, weights, bias, upper, lower)
testDense!(xInput, weights, bias,upper, lower, actFunc="")
