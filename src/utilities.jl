name = "utilities"
for file in readdir(joinpath(@__DIR__, name))
    endswith(file, ".jl") && include(joinpath(name, file))
end
