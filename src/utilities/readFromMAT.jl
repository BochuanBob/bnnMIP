using MAT
export readOneVar
# Read a single variable in the .mat file.
function readOneVar(fileName::String, varName::String)
    file = matopen(fileName)
    var = read(file, varName)
    close(file)
    return var
end
