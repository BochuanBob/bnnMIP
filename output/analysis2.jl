using CSV
using StatsBase

const FAMILIES = ["small", "basic", "extended", "extended2"]
const METHODS = ["NoCuts", "UserCuts", "DefaultCuts", "AllCuts", "UserCutsCover"]
const TIME_LIMIT = 1800.0

# less than 10 s
trivialList = []
# less than 100 s
easyList = []
# less than 1000 s
medianList = []
# otherwise
hardList = []




df = CSV.read("results3F200Ep008_2.csv")

function _mip_gap(row)
    return abs(row.Objs - row.Bounds) / (1e-10 + abs(row.Bounds))
end

df.gap = 0.0
df[!, :gap] .= 0.0
for i in 1:size(df, 1)
    df.gap[i] = _mip_gap(df[i,:])
end

for instance in unique(subset_df.Instance)
    slice = subset_df[subset_df.Instance .== instance, :]
    if size(slice, 1) == 0 continue end
    worst_solve_time = maximum(slice.RunTimes)
    if worst_solve_time < 10
        append!(trivialList, instance)
    elseif worst_solve_time < 100
        append!(easyList, instance)
    elseif worst_solve_time < 1000
        append!(medianList, instance)
    elseif worst_solve_time < TIME_LIMIT * 1.01
        append!(hardList, instance)
    end
end
levelList = [trivialList, easyList, medianList, hardList]
levelName = ["trivial", "easy", "medium", "hard"]
subset_df = df
total_instance_count = length(unique(subset_df.Instance))
winnerList = []
for level in levelList
    winners = Dict(method => 0 for method in METHODS)
    for instance in level
        slice = subset_df[subset_df.Instance .== instance, :]
        if size(slice, 1) == 0 continue end
        best_solve_time = minimum(slice.RunTimes)
        worst_solve_time = maximum(slice.RunTimes)

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
    push!(winnerList, winners)
end
for i in 1:4
    println("###########")
    println(levelName[i])
    winners = winnerList[i]
    for method in METHODS
        slice = subset_df[(subset_df.Methods .== method) .& in.(subset_df.Instance, Ref(levelList[i])), :]
        println("$method")
        println("-" ^ (length(method) + 2))
        println("  * solve time: ", mean(slice.RunTimes), " ± ", StatsBase.std(slice.RunTimes), " sec")
        println("  * MIP gap:    ", mean(slice.gap), " ± ", StatsBase.std(slice.gap))
        num_winners = winners[method]
        println("  * Winners:    ", num_winners, " / ", total_instance_count)
    end
end
