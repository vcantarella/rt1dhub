
using BenchmarkTools
using Plots
using FiniteDiff


module TransportAnalytical
using Zygote
using SpecialFunctions
"""
constant_injection(cr, x, t, c0, c_in, v, Dl)

Calculates the concentration profile of a solute undergoing
advection-dispersion transport in a porous 1D domain with constant
 in a 1D domain with constant injection
Original reference: (Lapidus and Amundson, 1952): 
Appelo, C.A.J.; Postma, Dieke. Geochemistry, Groundwater and Pollution (p. 104).

# Arguments
- `cr::Matrix`: A 2D array to store the concentration profile of a solute. 
    first dimension is time and second dimension is space.
- `x::Vector`: A 1D array of spatial locations. (Where to sample the concentration)
- `t::Vector`: A 1D array of time locations. (When to sample the concentration)
- `c0::Real`: The concentratio at x=0 (inflow concentration).
- `c_in::Real`: The initial concentration in the column (t=0).
- `v::Real`: The velocity of the solute.
- `Dl::Real`: The longitudinal dispersion coefficient.
# Returns
    nothing, the results are stored in the `cr` array. 
"""
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
        @inbounds cr[:, i] .= c_in .+ (c0 - c_in) / 2 .* (erfc.((x[i] .- v .* t)
         ./ (2 .* sqrt.(Dl .* t))) .+ exp(v .* x[i] / Dl)
          .* erfc.((x[i] .+ v .* t) ./ (2 .* sqrt.(Dl .* t))))
    end
    return nothing
end

"""
pulse_injection(cr, x, t, c0, c_in, v, Dl, t_pulse)

Calculates the concentration profile of a solute undergoing
advection-dispersion transport in a porous 1D domain with pulse injection
(starts at t=0 and ends at t=t_pulse)
Original reference: (Lapidus and Amundson, 1952):
Appelo, C.A.J.; Postma, Dieke. Geochemistry, Groundwater and Pollution (p. 104).

# Arguments
- `cr::Matrix`: A 2D array to store the concentration profile of a solute. 
    first dimension is time and second dimension is space.
- `x::Vector`: A 1D array of spatial locations. (Where to sample the concentration)
- `t::Vector`: A 1D array of time locations. (When to sample the concentration)
- `c0::Real`: The concentratio at x=0 (inflow concentration).
- `c_in::Real`: The initial concentration in the column (t=0).
- `v::Real`: The velocity of the solute.
- `Dl::Real`: The longitudinal dispersion coefficient.
- `t_pulse::Real`: The time at which the pulse injection ends (at x=0).
"""
function pulse_injection(
    cr::Union{Matrix{T}, Zygote.Buffer{T}} where T,
    x::Vector,
    t::Vector,
    c0::Number,
    c_in::Number,
    v::Number,
    Dl::Number,
    t_pulse::Number
    )
    #cr = zeros(length(t), length(x))
    ratio = (c0 - c_in) / 2
    tau = 0.0
    @inbounds for i=1:length(x), j=1:length(t)
        exp_term = exp(v .* x[i] / Dl)
        cr[j, i] = c_in .+ ratio .* (erfc.((x[i] .- v .* t[j])
         ./ (2 .* sqrt.(Dl .* t[j]))) .+ exp_term
         .* erfc.((x[i] .+ v .* t[j]) ./ (2 .* sqrt.(Dl .* t[j]))))
        if t[j] >= t_pulse
            tau = t_pulse
        else
            tau = 0.0
        end
        co = - ratio .* (erfc.((x[i] .- v .* (t[j] .- tau))
        ./ (2 .* sqrt.(Dl .* (t[j] .- tau)))) .+ exp_term
        .* erfc.((x[i] .+ v .* (t[j] .- tau)) ./ (2 .* sqrt.(Dl .* (t[j] .- tau)))))
        println(co)
        cr[j, i] = cr[j, i] .+ co
    end
    return nothing
end
end # module

# Main script code
using .TransportAnalytical
# Your main script code here
println("This script is being run directly.")
# Example usage
x = collect(range(0, stop=0.5, length=100))
t = collect(range(0.1, stop=72000, length=100))
c0 = 1.0
c_in = 0.0
v = 1e-5
alpha_l = 1e-3
Dl = 1e-9 + alpha_l * v
t_pulse = 3600.0
cr = zeros(length(t), length(x))
cr_pulse = zeros(length(t), length(x))


@btime TransportAnalytical.constant_injection(cr, collect(x), collect(t), c0, c_in, v, Dl)
@btime TransportAnalytical.pulse_injection(cr_pulse,collect(x), collect(t), c0, c_in, v, Dl, t_pulse)

using ForwardDiff

@btime fordiff0 = ForwardDiff.jacobian((cr, p)-> TransportAnalytical.constant_injection(cr, collect(x), collect(t), c0, c_in, p[1], p[2]),cr, [v, Dl])
@btime fordiff= ForwardDiff.jacobian((cr, p)-> TransportAnalytical.pulse_injection(cr, collect(x), collect(t), c0, c_in, p[1], p[2], t_pulse),cr_pulse, [v, Dl])

@btime finitdiff0 = FiniteDiff.finite_difference_derivative((p) -> begin
    TransportAnalytical.constant_injection(cr, collect(x), collect(t), c0, c_in, p[1], p[2])
    return cr
end, [v, Dl])

# Define a wrapper function for finite differences
function pulse_injection_wrapper(p)
    cr_pulse .= 0  # Reset cr_pulse to zero before each call
    TransportAnalytical.pulse_injection(cr_pulse, x, t, c0, c_in, p[1], p[2], t_pulse)
    return cr_pulse
end

@btime finitdiff = FiniteDiff.finite_difference_jacobian((p) -> begin
    TransportAnalytical.pulse_injection(cr_pulse, collect(x), collect(t), c0, c_in, p[1], p[2], t_pulse)
    return cr_pulse
end, [v, Dl])
finitdiff = FiniteDiff.finite_difference_jacobian((p) -> begin
    TransportAnalytical.pulse_injection(cr_pulse, collect(x), collect(t), c0, c_in, p[1], p[2], t_pulse)
    return cr_pulse
end, [v, Dl])
# function buffered(cr, p)
#     cr_ = Zygote.Buffer(cr)
#     TransportAnalytical.pulse_injection(cr_, collect(x), collect(t), c0, c_in, p[1], p[2], t_pulse)
#     return copy(cr_)
# end

# zygotediff = Zygote.jacobian((p)-> buffered(cr_pulse, p), [v, Dl])[1]

# @assert isapprox(fordiff, enzydiff, atol=1e-6)
# @assert isapprox(fordiff, zygotediff, atol=1e-6)
# @assert isapprox(enzydiff, zygotediff, atol=1e-6)


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
