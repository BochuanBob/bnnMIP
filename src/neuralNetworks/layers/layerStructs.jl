include("layerSetup.jl")

mutable struct FlattenLayer
    inputSize::NTuple{N, Int} where {N}
    xIn::VarOrAff
    xOut::VarOrAff
    FlattenLayer() = new((0,0), Array{VariableRef, 1}(undef, 0),
    Array{VariableRef, 1}(undef, 0))
end

mutable struct Conv2dLayer
    inputSize::NTuple{N, Int} where {N}
    weights::Array{Float64, 4}
    bias::Array{Float64, 1}
    upper::Array{Float64, 3}
    lower::Array{Float64, 3}
    strides::Tuple{Int64, Int64}
    padding::String
    activation::String
    tauList::Array{Float64, 1}
    kappaList::Array{Float64, 1}
    nonzeroIndicesList::Array{Array{CartesianIndex{3}, 1}, 3}
    uNewList::Array{Array{Float64, 1}, 3}
    lNewList::Array{Array{Float64, 1}, 3}
    xIn::VarOrAff
    xOut::VarOrAff
    Conv2dLayer() = new((0,0),Array{Float64, 4}(undef, (0,0,0,0)),
                    Array{Float64, 1}(undef, 0),
                    Array{Float64, 3}(undef, (0,0,0)),
                    Array{Float64, 3}(undef, (0,0,0)),
                    (0,0),
                    "",
                    "",
                    Array{Float64, 1}(undef, 0),
                    Array{Float64, 1}(undef, 0),
                    Array{Array{CartesianIndex{3}, 1}, 3}(undef, (0,0,0)),
                    Array{Array{Float64, 1}, 3}(undef, (0,0,0)),
                    Array{Array{Float64, 1}, 3}(undef, (0,0,0)),
                    Array{VariableRef, 1}(undef, 0),
                    Array{VariableRef, 1}(undef, 0)
                    )
end

mutable struct Conv2dBinLayer
    weights::Array{Float64, 4}
    bias::Array{Float64, 1}
    strides::Tuple{Int64, Int64}
    padding::String
    activation::String
    tauList::Array{Float64, 1}
    kappaList::Array{Float64, 1}
    oneIndicesList::Array{Array{CartesianIndex{3}, 1}, 3}
    negOneIndicesList::Array{Array{CartesianIndex{3}, 1}, 3}
    xIn::VarOrAff
    xOut::VarOrAff
    Conv2dBinLayer() = new(Array{Float64, 4}(undef, (0,0,0,0)),
                    Array{Float64, 1}(undef, 0),
                    (0,0),
                    "",
                    "",
                    Array{Float64, 1}(undef, 0),
                    Array{Float64, 1}(undef, 0),
                    Array{Array{CartesianIndex{3}, 1}, 3}(undef, (0,0,0)),
                    Array{Array{CartesianIndex{3}, 1}, 3}(undef, (0,0,0)),
                    Array{VariableRef, 1}(undef, 0),
                    Array{VariableRef, 1}(undef, 0)
                    )
end

mutable struct DenseLayer
    weights::Array{Float64, 2}
    bias::Array{Float64, 1}
    upper::Array{Float64, 1}
    lower::Array{Float64, 1}
    activation::String
    tauList::Array{Float64, 1}
    kappaList::Array{Float64, 1}
    nonzeroIndicesList::Array{Array{Int64, 1}}
    uNewList::Array{Array{Float64, 1}, 1}
    lNewList::Array{Array{Float64, 1}, 1}
    takeSign::Bool
    xIn::VarOrAff
    xOut::VarOrAff
    DenseLayer() = new(Array{Float64, 2}(undef, (0,0)),
                    Array{Float64, 1}(undef, 0),
                    Array{Float64, 1}(undef, 0),
                    Array{Float64, 1}(undef, 0),
                    "",
                    Array{Float64, 1}(undef, 0),
                    Array{Float64, 1}(undef, 0),
                    Array{Array{Int64, 1}, 1}(undef, 0),
                    Array{Array{Float64, 1}, 1}(undef, 0),
                    Array{Array{Float64, 1}, 1}(undef, 0),
                    false,
                    Array{VariableRef, 1}(undef, 0),
                    Array{VariableRef, 1}(undef, 0)
                    )
end

mutable struct DenseBinLayer
    weights::Array{Float64, 2}
    bias::Array{Float64, 1}
    activation::String
    tauList::Array{Float64, 1}
    kappaList::Array{Float64, 1}
    oneIndicesList::Array{Array{Int64, 1}, 1}
    negOneIndicesList::Array{Array{Int64, 1}, 1}
    takeSign::Bool
    xIn::VarOrAff
    xOut::VarOrAff
    DenseBinLayer() = new(Array{Float64, 2}(undef, (0,0)),
                    Array{Float64, 1}(undef, 0),
                    "",
                    Array{Float64, 1}(undef, 0),
                    Array{Float64, 1}(undef, 0),
                    Array{Array{Int64, 1}, 1}(undef, 0),
                    Array{Array{Int64, 1}, 1}(undef, 0),
                    false,
                    Array{VariableRef, 1}(undef, 0),
                    Array{VariableRef, 1}(undef, 0)
                    )
end

NNLayer = Union{FlattenLayer, Conv2dLayer, Conv2dBinLayer,
                DenseLayer, DenseBinLayer}