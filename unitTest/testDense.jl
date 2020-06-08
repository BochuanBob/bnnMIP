include("../src/neuralNetworks/layers/dense.jl")
using JuMP, Gurobi

# transformProc
upper = [1, 2, 3, 4]
lower = [0, -1, -2, -3]
negIndices = [1, 3]
@assert(transformProc(negIndices, upper, lower) ==
            ([0, 2, -2, 4], [1, -1, 3, -3]))
negIndices = Array{Int64, 1}([])
@assert(transformProc(negIndices, upper, lower) ==
            ([1, 2, 3, 4], [0, -1, -2, -3]))

# decideViolationCons, getFirstCutIndices, getSecondCutIndices
xVal = [-1, 1, 1, 1, -1]
yVal = 0
nonzeroIndices = [1, 2, 4, 5]
w = [-1, 1, 0, 1, -1]
tau, kappa = 0, 0
upper, lower = [1,1,1,1,1], [-1,-1,-1,-1,-1]
negIndices = findall(w .< 0)
uNew, lNew = transformProc(negIndices, upper, lower)
@assert(decideViolationCons(xVal, yVal, nonzeroIndices, w,
        tau, kappa, uNew, lNew) == (-4, 4))
@assert(sort(getSecondCutIndices(xVal, yVal, nonzeroIndices, w, uNew, lNew))
        == [1, 2, 4, 5])

w = [1, -1, 0, -1, 1]
negIndices = findall(w .< 0)
uNew, lNew = transformProc(negIndices, upper, lower)
@assert(decideViolationCons(xVal, yVal, nonzeroIndices, w,
        tau, kappa, uNew, lNew) == (4, -4))
@assert(sort(getFirstCutIndices(xVal, yVal, nonzeroIndices, w, uNew, lNew))
        == [1, 2, 4, 5])
yVal = -1
@assert(decideViolationCons(xVal, yVal, nonzeroIndices, w,
        tau, kappa, uNew, lNew) == (0, -8))

yVal = 1
@assert(decideViolationCons(xVal, yVal, nonzeroIndices, w,
        tau, kappa, uNew, lNew) == (8, 0))
@assert(sort(getFirstCutIndices(xVal, yVal, nonzeroIndices, w, uNew, lNew))
        == [1, 2, 4, 5])

# getFirstCon, getSecondCon
m = direct_model(Gurobi.Optimizer(OutputFlag=1))
x = @variable(m, [1:5], base_name="x")
@variable(m, y, base_name="y")
xVal = [-1, 1, 1, 1, -1]
yVal = 0
nonzeroIndices = [1, 2, 4, 5]
w = [-1, 1, 0, 1, -1]
tau, kappa = 0, 0
upper, lower = [1,1,1,1,1], [-1,-1,-1,-1,-1]
negIndices = findall(w .< 0)
uNew, lNew = transformProc(negIndices, upper, lower)
Iset = getSecondCutIndices(xVal, yVal, nonzeroIndices, w, uNew, lNew)
@show getSecondCon(x, y, Iset, nonzeroIndices, w, kappa, uNew, lNew)

w = [1, -1, 0, -1, 1]
negIndices = findall(w .< 0)
uNew, lNew = transformProc(negIndices, upper, lower)
Iset = getFirstCutIndices(xVal, yVal, nonzeroIndices, w, uNew, lNew)
@show getFirstCon(x, y, Iset, nonzeroIndices, w, tau, uNew, lNew)
