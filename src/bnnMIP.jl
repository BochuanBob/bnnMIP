module bnnMIP
    import JuMP, Gurobi
    import MAT
    import Combinatorics, IterTools

    using JuMP, Gurobi
    using MAT
    using Combinatorics, IterTools
    include("neuralNetworks.jl")
    include("utilities.jl")
    include("verification.jl")
end  # module bnnMIP
