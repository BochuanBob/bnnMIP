module bnnMIP
    import JuMP, Gurobi
    import MAT
    import Combinatorics, IterTools
    import Random

    using JuMP, Gurobi
    using MAT
    using Random
    using Combinatorics, IterTools
    const CUTOFF_DENSE_BIN = 100 # Not Used in Code
    const CUTOFF_DENSE_BIN_PRECUT = 10
    const NONZERO_MAX_DENSE_BIN = 50
    const EXTEND_CUTOFF_DENSE_BIN = 10

    const CUTOFF_DENSE = 10 # Not Used in Code
    const CUTOFF_DENSE_PRECUT = 5
    const NONZERO_MAX_DENSE = 50
    const EXTEND_CUTOFF_DENSE = 10

    const CUTOFF_CONV2D = 100
    const CUTOFF_CONV2D_PRECUT = 5

    const CUTOFF_CONV2D_BIN = 100
    const CUTOFF_CONV2D_BIN_PRECUT = 5

    export NoCuts, DefaultCuts, UserCuts, UserCutsCover, CoverCuts, AllCuts,
            NoCutsForward, DefaultCutsForward, UserCutsForward,
            UserCutsCoverForward, CoverCutsForward, AllCutsForward,
            bnnMIPparameters

    mutable struct bnnMIPparameters
        integer::Bool
        image::Bool
        useDense::Bool
        useConv2d::Bool
        consistDense::Bool
        consistDenseBin::Bool
        switchCuts::Bool
        K::Int64
        bnnMIPparameters() = new(false, true, false, false, true, true, false, 0)
    end

    struct Method
        userCuts::Bool
        preCuts::Bool
        forward::Bool
        coverCuts::Int64
        allCuts::Int64
    end

    const NoCuts = Method(false, false, false, 0, 0)
    const DefaultCuts = Method(false, false, false, 1, 1)
    const UserCuts = Method(true, false, false, 0, 0)
    const UserCutsCover = Method(true, false, false, 1, 0)
    const CoverCuts = Method(false, false, false, 1, 0)
    const AllCuts = Method(true, false, true, 1, 1)
    const NoCutsForward = Method(false, false, true, 0, 0)
    const DefaultCutsForward = Method(false, false, true, 1, 1)
    const UserCutsForward = Method(true, false, true, 0, 0)
    const UserCutsCoverForward = Method(true, false, true, 1, 0)
    const CoverCutsForward = Method(false, false, true, 1, 0)
    const AllCutsForward = Method(true, false, true, 1, 1)

    include("neuralNetworks.jl")
    include("utilities.jl")
    include("verification.jl")
end  # module bnnMIP
