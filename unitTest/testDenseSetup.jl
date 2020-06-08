include("../src/neuralNetworks/layers/denseSetup.jl")

# b - 1 and b + 1
@assert(getTauAndKappa(10, -1) == (-2, 0))
# floor(b) and ceil(b) + 1
@assert(getTauAndKappa(10, -1.4) == (-2, 0))
# b and b + 2
@assert(getTauAndKappa(9, -1) == (-1, 1))
# floor(b) - 1 and ceil(b)
@assert(getTauAndKappa(9, -1.4) == (-3, -1))

# b and b + 2
@assert(getTauAndKappa(100, 2) == (2, 4))
# floor(b) - 1 and ceil(b)
@assert(getTauAndKappa(100, 1.9) == (0,2))
# b - 1 and b + 1
@assert(getTauAndKappa(99, 2) == (1, 3))
# floor(b) and ceil(b) + 1
@assert(getTauAndKappa(99, 1.9) == (1, 3))
