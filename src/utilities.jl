name = "utilities"
include(joinpath(name, "preprocessNN.jl"))
for file in readdir(joinpath(@__DIR__, name))
    if (file != "preprocessNN.jl")
        endswith(file, ".jl") && include(joinpath(name, file))
    end
end
