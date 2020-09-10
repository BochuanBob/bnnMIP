using JuMP, Gurobi
using MAT
using Random
using Combinatorics, IterTools
const VarOrAff = Union{JuMP.VariableRef,JuMP.AffExpr}

include("denseLP.jl")
include("flatten.jl")
include("utilities.jl")
include("verification.jl")
