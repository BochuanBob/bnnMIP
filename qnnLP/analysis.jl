using CSV
using StatsBase

io = open("qnnOutput/2F100EP005summarySparseAcc.txt", "w")

df = CSV.read("qnnOutput/2F100compareSparseAccEP005.csv")
for model in unique(df.Models)
    sub_df = df[df.Models .== model, :]
    write(io, string("Model: ", model, " Verified: ", sum(sub_df.Objs .>= 0),
            " Out of: ", length(sub_df.Objs), "\n"))
end

close(io)
