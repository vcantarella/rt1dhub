"""
This test the functions for the analytical solution of the transport equation
and the pulse injection function. We expect the python and julia versions to throw the same results.
We import julia funtions and run the tests. Python code is imported using PythonCall.jl
"""

using PythonCall
using Test
using BenchmarkTools
using DifferentialEquations
using Plots
using LinearAlgebra
include("../src/transport_analytical.jl")
include("../src/ODE_transport.jl")



# Your main script code here
println("This script is being run directly.")
# Example usage
x = collect(range(0, stop=0.5, length=200))
ts = collect(range(0.1, stop=72000, length=100))
c0 = 1.0
c_in = 0.0
v = 1e-5
alpha_l = 1e-2
Dl = 1e-9 + alpha_l * v
t_pulse = 3600.0
cr = zeros(length(ts), length(x))
cr_pulse = zeros(length(ts), length(x))
ϕ = 0.3
q = v * ϕ
ADEST! = create_ADEST(x[2]-x[1], 1e-9, q)
ADE_pulse! = create_ADEST_pulse(x[2]-x[1], 1e-9, q, t_pulse)

np = pyimport("numpy")
sp = pyimport("scipy.special")
# Analytical ogata_banks in python:
function constant_injection(
    cr, x, t, c0, c_in, v,Dl)
    #cr = np.zeros((t.shape[0], x.shape[0]))
    for i in eachindex(x)
        cr[:, i] = c_in + (c0 - c_in) / 2 * (
            scipy.special.erfc((x[i] - v * t) / (2 * np.sqrt(Dl * t)))
        )
    end
end

cr_py = zeros(length(ts), length(x))

constant_injection(cr, x, ts, c0, c_in, v, Dl)

constant_injection(cr, collect(x), collect(ts), c0, c_in, v, Dl)
# Running the numerical integration with an ODE
u0 = zeros(length(x)).+c_in
p = [ϕ, alpha_l, c0]
prob = ODEProblem(ADEST!, u0, (0.0, 72000.0), p)
sol = solve(prob, Rosenbrock23(), saveat=ts, abstol=1e-10, reltol=1e-10)
# plotting the results
for i in 40:length(ts)
    @test isapprox(sol.u[i], cr[i, :], rtol = 5e-2)
end
# Plot both solutions
p = plot()
colors = range(0, stop=1, length=length(ts))  # Create a range of colors
for (i, ti) in enumerate(ts)
    scatter!(p, x[1:10:length(x)], cr[i, 1:10:length(x)], label="", color=cgrad(:viridis)[colors[i]], linestyle=:dash, colorbar = false)
    plot!(p, x, sol.u[i], label="", color=cgrad(:viridis)[colors[i]], linestyle=:dash, colorbar = false)
end
# Add a dummy plot to create the colorbar with custom ticks and labels
plot!(p, x[1:10:length(x)], cr[1, 1:10:length(x)], label="", c=:viridis, colorbar_title="cbar", colorbar = true)
xlabel!("Distance x [m]")
ylabel!("Normalized Concentration")
title!("Concentration vs Distance for Different Times")
display(p)



pulse_injection(cr_pulse,collect(x), collect(ts), c0, c_in, v, Dl, t_pulse)

# Running the numerical integration with an ODE
u0 = zeros(length(x)).+c_in
p = [ϕ, alpha_l, c0]
prob = ODEProblem(ADEST!, u0, (0.0, 72000.0), p)
condition(u,t,integrator) = t >= t_pulse
affect!(integrator) = integrator.p[3] = c_in
callback = DiscreteCallback(condition, affect!)
sol = solve(prob, Tsit5(), saveat=ts, callback = callback, abstol=1e-11, reltol=1e-11)
# plotting the results
p_pulse = [ϕ, alpha_l, c0]
prob_pulse = ODEProblem(ADE_pulse!, u0, (0.0, 72000.0), p_pulse)
sol_pulse = solve(prob_pulse, Tsit5(), saveat=ts, abstol=1e-11, reltol=1e-11, tstops = [t_pulse])
@test sol_pulse.t == ts
p = plot()
colors = range(0, stop=1, length=length(ts))  # Create a range of colors


for (i, ti) in enumerate(ts)
    tt = ts[i]
    sol_index = findfirst(sol.t .== tt)
    scatter!(p, x[1:10:length(x)], cr_pulse[i, 1:10:length(x)], label="", color=cgrad(:viridis)[colors[i]], linestyle=:dash, colorbar = false)
    plot!(p, x, sol.u[sol_index], label="", color=cgrad(:viridis)[colors[i]], linestyle=:dash, colorbar = false)
    plot!(p, x, sol_pulse.u[i], label="", color=cgrad(:viridis)[colors[i]], linestyle=:dash, colorbar = false)
end

# Add a dummy plot to create the colorbar with custom ticks and labels
plot!(p, x, sol.u[1], label="", c=:viridis, colorbar_title="cbar", colorbar = true)

xlabel!("Distance x [m]")
ylabel!("Normalized Concentration")
title!("Concentration vs Distance for Different Times")
display(p)


p = plot()
colors = range(0, stop=1, length=length(sol.t))  # Create a range of colors


for (i, ti) in enumerate(sol.t)
    plot!(p, x, sol.u[i], label="", color=cgrad(:viridis)[colors[i]], linestyle=:dash, colorbar = false)
end

# Add a dummy plot to create the colorbar with custom ticks and labels
plot!(p, x, sol.u[1], label="", c=:viridis, colorbar_title="cbar", colorbar = true)

xlabel!("Distance x [m]")
ylabel!("Normalized Concentration")
title!("Concentration vs Distance for Different Times")
display(p)

# Check discrepance between the analytical and numerical solutions
norm_mean_squared_error(a, b) = sqrt(sum((a-b).^2)/length(a))/(maximum(a)-minimum(a))
## check first ogata banks > 0 solution (valid range)
findfirst(cr_pulse[:,1].> 0)
for i in 6:length(ts)
    tt = ts[i]
    local sol_index = findlast(sol.t .== tt)
    @test sol.t[sol_index] == ts[i]
    @test sol.t[sol_index] == sol_pulse.t[i]
    try
        @test norm_mean_squared_error(cr_pulse[i, :], sol_pulse.u[i]) < 5e-2
        @test norm_mean_squared_error(cr_pulse[i, :], sol.u[sol_index]) < 6e-2
        @test norm_mean_squared_error(cr_pulse[i, :], sol_pulse.u[i]) < 5e-2
    catch
        println("Error at time: ", tt)
    end
end

for i in 1:length(sol.t)-1
    if sol.t[i+1] == sol.t[i]
        @test sol.u[i+1] == sol.u[i]
    end
end

# Plot the concentration at a given time
t_p = ts[50]
p = plot()
colors = range(0, stop=1, length=length(ts))  # Create a range of colors
tt_index = findfirst(sol_pulse.t .== t_p)
sol_index = findfirst(sol.t .== t_p)
plot!(p, x, sol.u[sol_index], label="", color=cgrad(:viridis)[colors[tt_index]], linestyle=:dash, colorbar = false)
plot!(p, x, sol_pulse.u[tt_index], label="", color=cgrad(:viridis)[colors[tt_index]], colorbar = false)
scatter!(p, x[1:10:length(x)], cr_pulse[tt_index, 1:10:length(x)], label="", color=cgrad(:viridis)[colors[tt_index]], colorbar = false)
xlabel!("Distance x [m]")
ylabel!("Normalized Concentration")
title!("Concentration vs Distance for Different Times")
display(p)