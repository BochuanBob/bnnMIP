using CSV
using StatsBase

io = open("EP01summarySparse.txt", "w")

df = CSV.read("compareSparseEP01.csv")
for model in unique(df.Models)
    sub_df = df[df.Models .== model, :]
    write(io, string("Model: ", model, " Verified: ", sum(sub_df.Objs .>= 0),
            " Out of: ", length(sub_df.Objs), "\n"))
end

close(io)
