include("../src/neuralNetworks/layers/denseBin.jl")
using JuMP, Gurobi
# checkWeights
mat = [[1 -1 0]; [1 1 1]]
@assert(checkWeights(mat))
mat = [[1.0 -1.0000 0]; [1.00000 1.000 1]]
@assert(checkWeights(mat))
mat = [[1.0 -1.00001 0]; [1.00000 1.000 1]]
@assert(~checkWeights(mat))
mat = [[1.0 -2 0]; [3 0 1]]
@assert(~checkWeights(mat))

# decideViolationConsBin, getFirstBinCutIndices, getSecondBinCutIndices
xVal = [-1, 1, 1, 1, -1]
yVal = 0
oneIndices = [2, 4]
negIndices = [1, 5]
tau, kappa = 0, 0
# Positive values mean the violations.
@assert(decideViolationConsBin(xVal, yVal, oneIndices,
                negIndices, tau, kappa) == (-2, 2))
Ipos, Ineg = getSecondBinCutIndices(xVal, yVal, oneIndices,
                negIndices)
@assert(sort(union(Ipos, Ineg)) == [1,2,4,5])
oneIndices = [1, 5]
negIndices = [2, 4]
@assert(decideViolationConsBin(xVal, yVal, oneIndices,
                negIndices, tau, kappa) == (2, -2))
Ipos, Ineg = getFirstBinCutIndices(xVal, yVal, oneIndices,
                negIndices)
@assert(sort(union(Ipos, Ineg)) == [1,2,4,5])
yVal = -1
@assert(decideViolationConsBin(xVal, yVal, oneIndices,
                negIndices, tau, kappa) == (0, -4))
yVal = 1
@assert(decideViolationConsBin(xVal, yVal, oneIndices,
                negIndices, tau, kappa) == (4, 0))
Ipos, Ineg = getFirstBinCutIndices(xVal, yVal, oneIndices,
                negIndices)
@assert(sort(union(Ipos, Ineg)) == [1,2,4,5])

# getFirstBinCon, getSecondBinCon
m = direct_model(Gurobi.Optimizer(OutputFlag=1))
x = @variable(m, [1:5], base_name="x")
@variable(m, y, base_name="y")
xVal = [-1, 1, 1, 1, -1]
yVal = 0
oneIndices = [2, 4]
negIndices = [1, 5]
nonzeroNum = 4
tau, kappa = 0, 0
Ipos, Ineg = getSecondBinCutIndices(xVal, yVal, oneIndices,
                negIndices)
lenI = length(Ipos) + length(Ineg)
@show getSecondBinCon(x, y, Ipos, Ineg, lenI, nonzeroNum, kappa)

oneIndices = [1, 5]
negIndices = [2, 4]
Ipos, Ineg = getFirstBinCutIndices(xVal, yVal, oneIndices,
                negIndices)
lenI = length(Ipos) + length(Ineg)
@show getFirstBinCon(x, y, Ipos, Ineg, lenI, nonzeroNum, tau)
