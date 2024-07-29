using SpecialFunctions
using Plots

function constant_injection(
     cr::Matrix{Float64},
     x::Vector{Float64},
     t::Vector{Float64},
     c0::Float64,
     c_in::Float64,
     v::Float64,
     Dl::Float64
     )
    
    @simd for i in 1:length(x)
        @inbounds cr[:, i] .= c_in .+ (c0 - c_in) / 2 .* (erfc.((x[i] .- v .* t)
         ./ (2 .* sqrt.(Dl .* t))) .+ exp(v .* x[i] / Dl)
          .* erfc.((x[i] .+ v .* t) ./ (2 .* sqrt.(Dl .* t))))
    end
    return nothing
end

function pulse_injection(
    cr::Matrix{Float64},
    x::Vector{Float64},
    t::Vector{Float64},
    c0::Float64,
    c_in::Float64,
    v::Float64,
    Dl::Float64,
    t_pulse::Float64
    )
    tau = zeros(length(t))
    tau[t .>= t_pulse] .= t_pulse
    #cr = zeros(length(t), length(x))
    @simd for i in 1:length(x)
        @inbounds cr[:, i] .= c_in .+ (c0 - c_in) / 2 .* (erfc.((x[i] .- v .* t)
         ./ (2 .* sqrt.(Dl .* t))) .+ exp(v .* x[i] / Dl)
         .* erfc.((x[i] .+ v .* t) ./ (2 .* sqrt.(Dl .* t))))
        co = - (c0 - c_in) / 2 .* (erfc.((x[i] .- v .* (t .- tau))
         ./ (2 .* sqrt.(Dl .* (t .- tau)))) .+ exp(v .* x[i] / Dl)
          .* erfc.((x[i] .+ v .* (t .- tau)) ./ (2 .* sqrt.(Dl .* (t .- tau)))))
        @inbounds cr[:, i] .+= co
    end
    return nothing
end

# Example usage
x = range(0, stop=0.5, length=100)
t = range(0.1, stop=72000, length=100)
c0 = 1.0
c_in = 0.0
v = 1e-5
alpha_l = 1e-3
Dl = 1e-9 + alpha_l * v
t_pulse = 3600.0
cr = zeros(length(t), length(x))
cr_pulse = zeros(length(t), length(x))
constant_injection(cr, collect(x), collect(t), c0, c_in, v, Dl)
pulse_injection(cr_pulse,collect(x), collect(t), c0, c_in, v, Dl, t_pulse)


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