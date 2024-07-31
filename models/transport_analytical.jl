using SpecialFunctions
using BenchmarkTools
using Plots

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

function pulse_injection(
    cr::Matrix,
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
    @inbounds for i=1:length(x), j=1:length(t)
        exp_term = exp(v .* x[i] / Dl)
        cr[j, i] = c_in .+ ratio .* (erfc.((x[i] .- v .* t[j])
         ./ (2 .* sqrt.(Dl .* t[j]))) .+ exp_term
         .* erfc.((x[i] .+ v .* t[j]) ./ (2 .* sqrt.(Dl .* t[j]))))
        if t[j] >= t_pulse
            co = - ratio .* (erfc.((x[i] .- v .* (t[j] .- t_pulse))
            ./ (2 .* sqrt.(Dl .* (t[j] .- t_pulse)))) .+ exp_term
            .* erfc.((x[i] .+ v .* (t[j] .- t_pulse)) ./ (2 .* sqrt.(Dl .* (t[j] .- t_pulse)))))
            cr[j, i] = cr[j, i] .+ co
        end
    end
    return nothing
end

# Main script code
if abspath(PROGRAM_FILE) == @__FILE__
    # Your main script code here
    println("This script is being run directly.")
    # Example usage
    global x = collect(range(0, stop=0.5, length=100))
    global t = collect(range(0.1, stop=72000, length=100))
    global c0 = 1.0
    global c_in = 0.0
    v = 1e-5
    alpha_l = 1e-3
    Dl = 1e-9 + alpha_l * v
    global t_pulse = 3600.0
    cr = zeros(length(t), length(x))
    cr_pulse = zeros(length(t), length(x))
    function global_pulse!(cr, p)
        v = p[1]
        Dl = p[2]
        #cr = zeros(length(t), length(x))
        ratio = (c0 - c_in) / 2
        @inbounds for i=1:length(x), j=1:length(t)
            exp_term = exp(v .* x[i] / Dl)
            cr[j, i] = c_in .+ ratio .* (erfc.((x[i] .- v .* t[j])
                ./ (2 .* sqrt.(Dl .* t[j]))) .+ exp_term
                .* erfc.((x[i] .+ v .* t[j]) ./ (2 .* sqrt.(Dl .* t[j]))))
            if t[j] >= t_pulse
                co = - ratio .* (erfc.((x[i] .- v .* (t[j] .- t_pulse))
                ./ (2 .* sqrt.(Dl .* (t[j] .- t_pulse)))) .+ exp_term
                .* erfc.((x[i] .+ v .* (t[j] .- t_pulse)) ./ (2 .* sqrt.(Dl .* (t[j] .- t_pulse)))))
                cr[j, i] = cr[j, i] .+ co
            end
        end
            return nothing
        end
    Threads.nthreads()
    @btime constant_injection(cr, collect(x), collect(t), c0, c_in, v, Dl)
    @btime pulse_injection(cr_pulse,collect(x), collect(t), c0, c_in, v, Dl, t_pulse)
    @btime global_pulse!(cr_pulse, [v, Dl])
    using ForwardDiff
    ForwardDiff.jacobian((cr, p)-> pulse_injection(cr, collect(x), collect(t), c0, c_in, p[1], p[2], t_pulse),cr_pulse, [v, Dl])

    ForwardDiff.jacobian(global_pulse!,cr_pulse, [v, Dl])
    
    using Enzyme
    dp = zeros(2)
    dp[1] = 1
    dcr = zeros(length(t), length(x))
    Enzyme.autodiff(Forward, (p, cr)-> pulse_injection(cr, collect(x), collect(t), c0, c_in, p[1], p[2], t_pulse), Duplicated([v, Dl], dp), Duplicated(cr, dcr));
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
end