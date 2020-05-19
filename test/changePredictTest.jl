using JuMP, Gurobi
include("testFunc.jl")
include("../src/utilities.jl")
include("../src/verification.jl")
nn = readNN("../data/nn.mat", "nn")
testImages = readOneVar("../data/data.mat", "test_images")
testLabels = readOneVar("../data/data.mat", "test_labels")
testLabels = Array{Int64, 1}(testLabels[:]) .+ 1

for cuts in [true]
    if (cuts)
        m = direct_model(Gurobi.Optimizer(OutputFlag=1, Cuts=0,
            CliqueCuts=0, CoverCuts=0, FlowCoverCuts=0,
            FlowPathCuts=0, GUBCoverCuts=0, ImpliedCuts=0,
            InfProofCuts=0, MIPSepCuts=0, MIRCuts=0,
            ModKCuts=0,NetworkCuts=0,ProjImpliedCuts=0,
            StrongCGCuts=0, SubMIPCuts=0, ZeroHalfCuts=0))
            set_optimizer_attribute(m, "GomoryPasses", 0)
            set_optimizer_attribute(m, "CutPasses", 2000000000)
    else
        m = direct_model(Gurobi.Optimizer(OutputFlag=1))
        set_optimizer_attribute(m, "CutPasses", 2000000000)
    end
    i = 1
    input=testImages[i,:,:,:]
    trueIndex=testLabels[i]
    x, y = falsePredictVerify(m, nn, input, trueIndex, cuts=cuts)
    println("Using our cuts: ", cuts)
    @time optimize!(m)
    println("Original: ", forwardProp(input))
    println("Output: ", value.(y))
end
