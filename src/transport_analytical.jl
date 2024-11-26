
using BenchmarkTools
using Plots
using FiniteDiff
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
        @inbounds cr[:, i] .= c_in .+ (c0 - c_in) / 2 .* erfc.((x[i] .- v .* t)
         ./ (2 .* sqrt.(Dl .* t))) #.+ exp(v .* x[i] / Dl)
         # .* erfc.((x[i] .+ v .* t) ./ (2 .* sqrt.(Dl .* t))))
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
        cr[j, i] = cr[j, i] .+ co
    end
    return nothing
end

