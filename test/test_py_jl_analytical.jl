"""
This test the functions for the analytical solution of the transport equation
and the pulse injection function. We expect the python and julia versions to throw the same results.
We import julia funtions and run the tests. Python code is imported using PythonCall.jl
"""

using PythonCall
using Test
using BenchmarkTools
using DifferentialEquations
include("../src/transport_analytical.jl")
include("../src/ODE_transport.jl")



# Your main script code here
println("This script is being run directly.")
# Example usage
x = collect(range(0, stop=0.5, length=200))
t = collect(range(0.1, stop=72000, length=100))
c0 = 1.0
c_in = 0.0
v = 1e-5
alpha_l = 1e-3
Dl = 1e-9 + alpha_l * v
t_pulse = 3600.0
cr = zeros(length(t), length(x))
cr_pulse = zeros(length(t), length(x))
ϕ = 0.3
q = v * ϕ
ADEST! = create_ADEST(x[2]-x[1], 1e-9, q)

constant_injection(cr, collect(x), collect(t), c0, c_in, v, Dl)
# Running the numerical integration with an ODE
u0 = zeros(length(x)).+c_in
p = [ϕ, alpha_l, c0]
prob = ODEProblem(ADEST!, u0, (0.0, 72000.0), p)
sol = solve(prob, Tsit5(), saveat=t, abstol=1e-8, reltol=1e-8)
# plotting the results
for i in 40:length(t)
    @test isapprox(sol.u[i], cr[i, :], rtol = 1e-2)
end
# Plot both solutions
p = plot()
colors = range(0, stop=1, length=length(t))  # Create a range of colors
for (i, ti) in enumerate(t)
    plot!(p, x, cr[i, :], label="", color=cgrad(:viridis)[colors[i]], linestyle=:dash, colorbar = false)
end
# Add a dummy plot to create the colorbar with custom ticks and labels
plot!(p, x, cr[1, :], label="", c=:viridis, colorbar_title="cbar", colorbar = true)
xlabel!("Distance x [m]")
ylabel!("Normalized Concentration")
title!("Concentration vs Distance for Different Times")
display(p)
p = plot()
colors = range(0, stop=1, length=length(t))  # Create a range of colors
for (i, ti) in enumerate(t)
    plot!(p, x, sol.u[i], label="", color=cgrad(:viridis)[colors[i]], linestyle=:dash, colorbar = false)
end
# Add a dummy plot to create the colorbar with custom ticks and labels
plot!(p, x, sol.u[1], label="", c=:viridis, colorbar_title="cbar", colorbar = true)
xlabel!("Distance x [m]")
ylabel!("Normalized Concentration")
title!("Concentration vs Distance for Different Times")
display(p)




pulse_injection(cr_pulse,collect(x), collect(t), c0, c_in, v, Dl, t_pulse)

# Running the numerical integration with an ODE
u0 = zeros(length(x)).+c_in
p = [ϕ, alpha_l, c0]
prob = ODEProblem(ADEST!, u0, (0.0, 72000.0), p)
condition(u,t,integrator) = t >= t_pulse
affect!(integrator) = integrator.p[3] = c_in
callback = DiscreteCallback(condition, affect!)
sol = solve(prob, Tsit5(), saveat=t, callback = callback, abstol=1e-8, reltol=1e-8)
# plotting the results

p = plot()
colors = range(0, stop=1, length=length(t))  # Create a range of colors


for (i, ti) in enumerate(t)
    plot!(p, x, cr_pulse[i, :], label="", color=cgrad(:viridis)[colors[i]], linestyle=:dash, colorbar = false)
end

# Add a dummy plot to create the colorbar with custom ticks and labels
plot!(p, x, cr_pulse[1, :], label="", c=:viridis, colorbar_title="cbar", colorbar = true)

xlabel!("Distance x [m]")
ylabel!("Normalized Concentration")
title!("Concentration vs Distance for Different Times")
display(p)


p = plot()
colors = range(0, stop=1, length=length(t))  # Create a range of colors


for (i, ti) in enumerate(t)
    plot!(p, x, sol.u[i], label="", color=cgrad(:viridis)[colors[i]], linestyle=:dash, colorbar = false)
end

# Add a dummy plot to create the colorbar with custom ticks and labels
plot!(p, x, sol.u[1], label="", c=:viridis, colorbar_title="cbar", colorbar = true)

xlabel!("Distance x [m]")
ylabel!("Normalized Concentration")
title!("Concentration vs Distance for Different Times")
display(p)

# Check discrpance between the analytical and numerical solutions
#@test 
for i in eachindex(t)
    tt = t[i]
    sol_index = findfirst(sol.t .== tt)
    @test isapprox(cr_pulse[i, :], sol.u[sol_index], atol=1e-3)
end
isapprox(sol.u, cr_pulse, atol=1e-3)