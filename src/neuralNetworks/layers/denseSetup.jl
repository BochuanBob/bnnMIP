export getTauAndKappa, getTauAndKappaInt, getTauAndKappaFloat, checkWeights

# Checking each entry of weights must be in -1, 0, 1.
function checkWeights(weights::Array{T, N}) where{T <: Real, N}
    for weight in weights
        if (~(weight in [-1, 0, 1]))
            return false
        end
    end
    return true
end

# When w^T x + kappa <= 0, sign(w^T x + b) = -1.
# When w^T x + tau >= 0, sign(w^T x + b) = 1.
# Each entry of w must be in -1, 0, 1.
# Return: tau, kappa.
function getTauAndKappa(nonzeroNum::Int64, b::U) where {U<:Real}
    if (floor(b) == ceil(b))
        return getTauAndKappaInt(nonzeroNum, Int64(b))
    else
        return getTauAndKappaFloat(nonzeroNum, Float64(b))
    end
end

# Get tau and kappa when b is an integer.
# Return: tau, kappa.
function getTauAndKappaInt(nonzeroNum::Int64, b::Int64)
    tau = Inf
    kappa = Inf
    if (mod(b, 2) == 0)
        if(mod(nonzeroNum, 2) == 0)
            tau = b
            kappa = b + 2
        else
            tau = b - 1
            kappa = b + 1
        end
    else
        if(mod(nonzeroNum, 2) == 0)
            tau = b - 1
            kappa = b + 1
        else
            tau = b
            kappa = b + 2
        end
    end
    return Int64(tau), Int64(kappa)
end

# Get tau and kappa when b is a real number but not an integer.
# Return: tau, kappa.
function getTauAndKappaFloat(nonzeroNum::Int64, b::Float64)
    tau = Inf
    kappa = Inf
    if (floor(b) == ceil(b))
        error("b can not be an integer!")
    end
    if (mod(floor(b), 2) == 0)
        if(mod(nonzeroNum, 2) == 0)
            tau = floor(b)
            kappa = ceil(b) + 1
        else
            tau = floor(b) - 1
            kappa = ceil(b)
        end
    else
        if(mod(nonzeroNum, 2) == 0)
            tau = floor(b) - 1
            kappa = ceil(b)
        else
            tau = floor(b)
            kappa = ceil(b) + 1
        end
    end
    return Int64(tau), Int64(kappa)
end
