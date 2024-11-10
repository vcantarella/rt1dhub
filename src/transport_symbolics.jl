
using BenchmarkTools
using Plots
using Symbolics
using SpecialFunctions
import IfElse


@variables c0 c_in v Dl x t
@variables q, ϕ #substitute variables (flow rate and porosity)
@variables αₗ, Dₑ   #substitute variable (dispersivity, eff. diffusion coefficient)

expr = c_in + (c0 - c_in) / 2 * (erfc((x - v * t)/
 (2 * sqrt(Dl * t))) + exp(v * x / Dl) * erfc((x + v * t) / (2 * sqrt(Dl * t))))

expr = substitute(expr, (Dict(Dl => Dₑ + αₗ * v)))
expr = substitute(expr, (Dict(v => q / ϕ)))
Dalpha = Differential(αₗ)
Dphi = Differential(ϕ)
expand_derivatives(Dalpha(expr))
expand_derivatives(Dphi(expr))

func = build_function(expr, [x, t, c0, c_in, q, Dₑ], [ϕ, αₗ])
f = eval(func)
f([0.4, 10200, 1.0, 0.1, 1e-5, 1e-9], [0.3, 1e-2])
@variables t_pulse
tau = IfElse.ifelse(t >= t_pulse, t_pulse, 0)

expr_pulse = expr - (c0 - c_in) / 2 * (erfc((x - v * (t-tau))/
 (2 * sqrt(Dl * (t-tau)))) + exp(v * x / Dl) * erfc((x + v * (t-tau)) / (2 * sqrt(Dl * (t-tau)))))
expr_pulse = substitute(expr_pulse, (Dict(Dl => Dₑ + αₗ * v)))
expr_pulse = substitute(expr_pulse, (Dict(v => q / ϕ)))

expand_derivatives(Dalpha(expr_pulse))
expand_derivatives(Dphi(expr_pulse))

func_pulse = build_function(simplify(expr_pulse), [x, t, c0, c_in, q, Dₑ, t_pulse], [ϕ, αₗ])
f = eval(func_pulse)
f([0.4, 10200, 1.0, 0.1, 1e-5, 1e-9, 3600], [0.3, 1e-2])
jac_Dalpha = build_function(expand_derivatives(Dalpha(expr_pulse)), [x, t, c0, c_in, q, Dₑ, t_pulse], [ϕ, αₗ])
jac_Dphi = build_function(expand_derivatives(Dphi(expr_pulse)), [x, t, c0, c_in, q, Dₑ, t_pulse], [ϕ, αₗ])
jac_Dalpha = eval(jac_Dalpha)
jac_Dphi = eval(jac_Dphi)
jac_Dalpha([0.4, 10200, 1.0, 0.1, 1e-5, 1e-9, 3600], [0.3, 1e-2])
x = collect(range(0, stop=0.5, length=10))
t = collect(range(0.1, stop=72000, length=100))

function vec_pulse!(cr, x, t, c0, c_in, q, Dₑ, t_pulse, ϕ, αₗ)
    for i in eachindex(t), j in eachindex(x)
        cr[j, i] = f([x[j], t[i], c0, c_in, q, Dₑ, t_pulse], [ϕ, αₗ])
    end
end

function jac_pulse!(jac, x, t, c0, c_in, q, Dₑ, t_pulse, ϕ, αₗ)
    for i in eachindex(t), j in eachindex(x)
        jac[j, i, 1] = jac_Dalpha([x[j], t[i], c0, c_in, q, Dₑ, t_pulse], [ϕ, αₗ])
        jac[j, i, 2] = jac_Dphi([x[j], t[i], c0, c_in, q, Dₑ, t_pulse], [ϕ, αₗ])
    end
end

# Example usage
x = collect(range(0.01, stop=0.5, length=40))
t = collect(range(0.1, stop=72000, length=100))
c0 = 1.0
c_in = 0.0
v = 1e-5
alpha_l = 1e-3
De = 1e-9
Dl = 1e-9 + alpha_l * v
t_pulse = 3600
phi = 0.3
q = 5e-6
cr = zeros(length(x), length(t))
jac = zeros(length(x), length(t), 2)


bench1 = @benchmark vec_pulse!(cr, collect(x), collect(t), c0, c_in, q, De, t_pulse, phi, alpha_l)
bench2 = @benchmark jac_pulse!(jac, collect(x), collect(t), c0, c_in, q, De, t_pulse, phi, alpha_l)



# plotting the results
p = plot()
colors = range(0, stop=1, length=length(t))  # Create a range of colors


for (i, ti) in enumerate(t)
    plot!(p, x, cr[:, i], label="", color=cgrad(:viridis)[colors[i]], linestyle=:dash, colorbar = false)
end

# Add a dummy plot to create the colorbar with custom ticks and labels
plot!(p, x, cr[:, 1], label="", c=:viridis, colorbar_title="cbar", colorbar = true)

xlabel!("Distance x [m]")
ylabel!("Normalized Concentration")
title!("Concentration vs Distance for Different Times")
display(p)
