include("layerSetup.jl")
export activation1D

function activation1D(m::JuMP.Model, x::VarOrAff,
	 				  funcName::String; upper=1, lower=-1)
    initNN!(m)
	count = m.ext[:NN].count
	m.ext[:NN].count += 1
	xLen = length(x)
	y = @variable(m, [1:xLen], base_name="y_$count")
	if (funcName=="relu")
		for i in 1:xLen
			relu!(m, x[i], y[i], upper=upper, lower=lower)
		end
	end
	return y
end

# relu for a single element.
function relu!(m::JuMP.Model, xi::VarOrAff, yi::VarOrAff;
			   upper=1, lower=-1)
	# TODO: Implement a relu function
	initNN!(m)
	count = m.ext[:NN].count
	m.ext[:NN].count += 1
	z = @variable(m, binary=true, base_name="z_$count")
	@constraint(m, yi <= xi - lower * (1 - z))
	@constraint(m, xi >= yi)
	@constraint(m, yi <= upper * z)
	@constraint(m, yi >= 0)
end
