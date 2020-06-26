name = "layers"

include(joinpath(name, "layerSetup.jl"))
include(joinpath(name, "denseSetup.jl"))
include(joinpath(name, "layerStructs.jl"))

for file in readdir(joinpath(@__DIR__, name))
    if (file != "layerSetup.jl" && file != "denseSetup.jl" &&
            file != "layerStructs.jl")
        endswith(file, ".jl") && include(joinpath(name, file))
    end
end
