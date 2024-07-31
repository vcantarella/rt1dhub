using SpecialFunctions
using BenchmarkTools
using Plots
using Optim
using OptimizationOptimJL
using Optimization
using SciMLSensitivity
using CSV
using DataFrames
using ForwardDiff
using NaNMath


function constant_injection(
     cr::Matrix,
     x::Vector,
     t::Vector,
     c0::Real,
     c_in::Real,
     v::Real,
     Dl::Real,
     )
    
    for i in 1:length(x)
        @inbounds cr[:, i] = c_in + (c0 - c_in) / 2 * (erfc((x[i] - v * t)
         / (2 * sqrt(Dl * t))) + exp(v * x[i] / Dl)
          * erfc((x[i] + v * t) / (2 * sqrt(Dl * t))))
    end
    return nothing
end

function pulse_injection!(
    cr::Matrix,
    x::Vector,
    t::Vector,
    c0,
    c_in,
    v,
    Dl,
    t_pulse
    ) where T
    #cr = zeros(length(t), length(x))
    ratio = (c0 - c_in) / 2
    @inbounds for i=1:length(x), j=1:length(t)
        exp_term = exp(v * x[i] / Dl)
        cr[j, i] = c_in + ratio * (erfc((x[i] - v * t[j])
         / (2 * NaNMath.sqrt(Dl * t[j]))) + exp_term
         * erfc((x[i] + v * t[j]) / (2 * NaNMath.sqrt(Dl * t[j]))))
        if t[j] > t_pulse
            co = - ratio * (erfc((x[i] - v * (t[j] - t_pulse))
            / (2 * NaNMath.sqrt(Dl * (t[j] - t_pulse)))) + exp_term
            * erfc((x[i] + v * (t[j] - t_pulse)) / (2 * NaNMath.sqrt(Dl * (t[j] - t_pulse)))))
            cr[j, i] = cr[j, i] + co
        end
    end
    return nothing
end

#load data
data = CSV.read("data/data_a.csv",DataFrame)
t = data[:,1]
c = data[:,2]
c0 = 2.0
c_in = 0.0

t_pulse = 3600.0
x = [0.121]
Q0_ml = 6 #ml/hr
Q0 = Q0_ml*1e-6/3600 # m3/s
diam = 0.037 # diameter of the column [m]
area = pi*(diam/2)^2 # cross-sectional area of the column [m2]
q = Q0/area # specific discharge [m/s]
De = 2.01e-9
c_model = zeros(length(t), length(x))
v_ = q/0.3
Dl_ = De + 1e-3*v_
# Plotting the primary model
using Plots
pulse_injection!(c_model, x, t, c0, c_in, v_, Dl_, t_pulse)
plot(t, c_model[:,1], label="", color=:green, linestyle=:dash, colorbar = false)


xlabel!("Distance x [m]")
ylabel!("Normalized Concentration")
title!("Concentration vs Distance for Different Times")
#display(p)



function loss_julia(p, x, t, c0, c_in, q, De, t_pulse)
    phi = p[1]
    alpha_l = p[2]
    vp = q / phi
    Dl = De + alpha_l * vp
    c_model = zeros(eltype(p),length(t), length(x))
    pulse_injection!(c_model, x, t, c0, c_in, vp, Dl, t_pulse)
    return sum((@view(c_model[:, 1]) - c).^2)
end


loss_julia([0.3, 1e-3], x, t, c0, c_in,q , De, t_pulse)

p0 = [0.3, 1e-3]
T = ForwardDiff.Dual
ForwardDiff.gradient((p)-> loss_julia(p, x, t, c0, c_in, q, De, t_pulse), p0)
using Enzyme
adtype = Optimization.AutoForwardDiff()
optf = Optimization.OptimizationFunction((u,p)->loss_julia(u, x, t, c0, c_in, q, De, t_pulse), adtype)
lb = [0.1, 1e-4]
ub = [0.7, 0.1]
p0 = [0.3, 1e-3]
optprob = Optimization.OptimizationProblem(optf, p0,)
result_ode = Optimization.solve(optprob, Optim.LBFGS(), abstol = 1e-8)

best_p = result_ode.u

# Plot model and compare with data:
t_m = collect(0.01:1:maximum(t))
c_model = zeros(length(t_m), length(x))
pulse_injection!(c_model, x, t_m, c0, c_in, q/best_p[1], De + best_p[2]*q/best_p[1], t_pulse)
p = plot(t_m, c_model[:,1], label="Model", color=:green, linestyle=:dash, colorbar = false)
scatter!(p, t, c, label="Data", color=:red)
xlabel!("time [s]")
ylabel!("Concentration [mmol/L]")
title!("Model vs Data")