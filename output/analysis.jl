using CSV
using StatsBase

const FAMILIES = ["small", "basic", "extended", "extended2"]
const METHODS = ["NoCuts", "UserCuts", "DefaultCuts", "AllCuts", "UserCutsCover"]
const TIME_LIMIT = 1800.0
const LOWER_TIME = 0.0

df = CSV.read("results2F500Ep008.csv")

function _mip_gap(row)
    return abs(row.Objs - row.Bounds) / (1e-10 + abs(row.Bounds))
end

df.gap = 0.0
df[!, :gap] .= 0.0
for i in 1:size(df, 1)
    df.gap[i] = _mip_gap(df[i,:])
end

subset_df = df
total_instance_count = length(unique(subset_df.Instance))

winners = Dict(method => 0 for method in METHODS)
for instance in unique(subset_df.Instance)
    slice = subset_df[subset_df.Instance .== instance, :]
    if size(slice, 1) == 0 continue end
    best_solve_time = minimum(slice.RunTimes)
    worst_solve_time = maximum(slice.RunTimes)
    if worst_solve_time < LOWER_TIME
        continue
    end
    if best_solve_time >= 0.99 * TIME_LIMIT continue end
    # NOTE: This excludes ties from the tally!
    if size(slice[slice.RunTimes .== best_solve_time, :], 1) != 1
        continue
    end
    for method in METHODS
        if minimum(slice[slice.Methods .== method, :RunTimes]) == best_solve_time
            winners[method] += 1
            # break
        end
    end
end
    for method in METHODS
    slice = subset_df[subset_df.Methods .== method, :]
    println("$method")
    println("-" ^ (length(method) + 2))
    println("  * solve time: ", mean(slice.RunTimes), " ± ", StatsBase.std(slice.RunTimes), " sec")
    println("  * MIP gap:    ", mean(slice.gap), " ± ", StatsBase.std(slice.gap))
    num_winners = winners[method]
    println("  * Winners:    ", num_winners, " / ", total_instance_count)
end
