include("layerSetup.jl")
export dense

# A fully connected layer with sign() or
# without activation function.
function dense(m::JuMP.Model, x::VarOrAff,
               weights::Array{T, 2}, bias::Array{U, 1};
               takeSign=false, cuts=true) where {T<:Real, U<:Real}
    initNN!(m)
    count = m.ext[:NN].count
    m.ext[:NN].count += 1
    (yLen, xLen) = size(weights)
    if (length(bias) != yLen)
        println("The sizes of weights and bias don't match!")
        exit()
    end
    y = @variable(m, [1:yLen], base_name="y_$count")
    if (takeSign)
        for i in 1:yLen
            neuronSign!(m, x, y[i], weights[i, :], bias[i], cuts=cuts)
        end
    else
        @constraint(m, [i=1:yLen], y[i] ==
                    bias[i] + sum(weights[i,j] * x[j] for j in 1:xLen))
    end
    return y
end

# A MIP formulation for a single neuron.
# If cuts == false, it is a Big-M formulation.
# Otherwise, cuts for the ideal formulation are added to the model.
function neuronSign!(m::JuMP.Model, x::VarOrAff, yi::VarOrAff,
                weightVec::Array{T, 1}, b::U;
                cuts=False) where {T<:Real, U<:Real}
    # TODO: Implement the MIP formulation.
    return nothing
end
