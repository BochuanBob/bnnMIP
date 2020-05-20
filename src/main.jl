using JuMP, Gurobi
include("utilities.jl")
include("verification.jl")
nn = readNN("../data/nn.mat", "nn")
testImages = readOneVar("../data/data.mat", "test_images")
testLabels = readOneVar("../data/data.mat", "test_labels")
testLabels = Array{Int64, 1}(testLabels[:]) .+ 1
# println(nn[2]["upper"])
# println(nn[2]["lower"])

for epsilon in [0.015]
    for cuts in [true]
        if (cuts)
            # m = direct_model(Gurobi.Optimizer(OutputFlag=1, Cuts=0,
            #     CliqueCuts=1, CoverCuts=1, FlowCoverCuts=1,
            #     FlowPathCuts=1, GUBCoverCuts=1, ImpliedCuts=1,
            #     InfProofCuts=1, MIPSepCuts=1, MIRCuts=1,
            #     ModKCuts=1,NetworkCuts=1,ProjImpliedCuts=1,
            #     StrongCGCuts=1, SubMIPCuts=1, ZeroHalfCuts=1))
            m = direct_model(Gurobi.Optimizer(OutputFlag=1, Cuts=1))
            # set_optimizer_attribute(m, "GomoryPasses", 0)
            # set_optimizer_attribute(m, "CutPasses", 2000000000)
        else
            m = direct_model(Gurobi.Optimizer(OutputFlag=1, Cuts=1))
            # set_optimizer_attribute(m, "CutPasses", 2000000000)
        end
        i = 1
        input=testImages[i,:,:,:]
        trueIndex=Int64(testLabels[i])
        targetIndex=mod(trueIndex + 1, 10) + 1
        x, y = perturbationVerify(m, nn, input, trueIndex,
                                targetIndex, epsilon, cuts=cuts)
        println("Using our cuts: ", cuts)
        println("Epsilon: ", epsilon)
        @time optimize!(m)
        println("Output: ", value.(y))
    end
end
