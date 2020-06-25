function my_callback_value(opt::Gurobi.Optimizer,
            cb_data::Gurobi.CallbackData, x::VariableRef)
    return my_get(
        opt, MOI.CallbackVariablePrimal(cb_data), index(x)
    )::Float64
end

function my_get(
    opt::Gurobi.Optimizer,
    ::MOI.CallbackVariablePrimal{CallbackData},
    x::MOI.VariableIndex
)
    return opt.callback_variable_primal[Gurobi._info(opt, x).column]::Float64
end
