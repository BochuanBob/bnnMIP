module bnnMIP
    import JuMP, Gurobi
    import MAT
    import Combinatorics, IterTools

    using JuMP, Gurobi
    using MAT
    using Combinatorics, IterTools
    const CUTOFF_DENSE_BIN = 100 # Not Used in Code
    const CUTOFF_DENSE_BIN_PRECUT = 10
    const NONZERO_MAX_DENSE_BIN = 1000
    const EXTEND_CUTOFF_DENSE_BIN = 10

    const CUTOFF_DENSE = 10 # Not Used in Code
    const CUTOFF_DENSE_PRECUT = 5
    const NONZERO_MAX_DENSE = 1000
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
        bnnMIPparameters() = new(false, true, false, false, true, true)
    end

    struct NoCuts
        userCuts::Bool
        preCuts::Bool
        forward::Bool
        coverCuts::Int64
        allCuts::Int64
        NoCuts() = new(false, false, false, 0, 0)
    end

    struct DefaultCuts
        userCuts::Bool
        preCuts::Bool
        forward::Bool
        coverCuts::Int64
        allCuts::Int64
        DefaultCuts() = new(false, false, false, 1, 1)
    end

    struct UserCuts
        userCuts::Bool
        preCuts::Bool
        forward::Bool
        coverCuts::Int64
        allCuts::Int64
        UserCuts() = new(true, false, false, 0, 0)
    end

    struct UserCutsCover
        userCuts::Bool
        preCuts::Bool
        forward::Bool
        coverCuts::Int64
        allCuts::Int64
        UserCutsCover() = new(true, false, false, 1, 0)
    end

    struct CoverCuts
        userCuts::Bool
        preCuts::Bool
        forward::Bool
        coverCuts::Int64
        allCuts::Int64
        CoverCuts() = new(false, false, false, 1, 0)
    end

    struct AllCuts
        userCuts::Bool
        preCuts::Bool
        forward::Bool
        coverCuts::Int64
        allCuts::Int64
        AllCuts() = new(true, false, true, 1, 1)
    end

    struct NoCutsForward
        userCuts::Bool
        preCuts::Bool
        forward::Bool
        coverCuts::Int64
        allCuts::Int64
        NoCutsForward() = new(false, false, true, 0, 0)
    end

    struct DefaultCutsForward
        userCuts::Bool
        preCuts::Bool
        forward::Bool
        coverCuts::Int64
        allCuts::Int64
        DefaultCutsForward() = new(false, false, true, 1, 1)
    end

    struct UserCutsForward
        userCuts::Bool
        preCuts::Bool
        forward::Bool
        coverCuts::Int64
        allCuts::Int64
        UserCutsForward() = new(true, false, true, 0, 0)
    end

    struct UserCutsCoverForward
        userCuts::Bool
        preCuts::Bool
        forward::Bool
        coverCuts::Int64
        allCuts::Int64
        UserCutsCoverForward() = new(true, false, true, 1, 0)
    end

    struct CoverCutsForward
        userCuts::Bool
        preCuts::Bool
        forward::Bool
        coverCuts::Int64
        allCuts::Int64
        CoverCutsForward() = new(false, false, true, 1, 0)
    end

    struct AllCutsForward
        userCuts::Bool
        preCuts::Bool
        forward::Bool
        coverCuts::Int64
        allCuts::Int64
        AllCutsForward() = new(true, false, true, 1, 1)
    end

    include("neuralNetworks.jl")
    include("utilities.jl")
    include("verification.jl")
end  # module bnnMIP
