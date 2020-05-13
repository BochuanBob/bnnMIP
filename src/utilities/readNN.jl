using MAT
include("preprocessNN.jl")

export readNN

# Read the information of the neural network and store it in desired format.
function readNN(fileName::String, nnName::String)
    file = matopen(fileName)
    nnPre = read(file, nnName)
    close(file)
    @assert haskey(nnPre[1], "inputSize")
    nn = preprocNN(nnPre)
    return nn
end
