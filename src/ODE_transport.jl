
"""
Create a right hand side (rhs) function for transport given the initial conditions.
The function is used to solve the transport equation using the DifferentialEquations.jl package.

# Arguments
- `Δx::Real`: The spatial step size.
- `cᵢ::Real`: The inflow concentration.
- `Dₑ::Real`: The effective difusion coefficient.
- `q::Real`: the specific flow rate.

# Returns
    A function that calculates the right hand side of the transport equation.

# Example
```julia
```
"""
function create_ADEST(Δx, Dₑ, q)
    function ADEST!(du, u, p ,t)
        ϕ = p[1] # porosity
        αₗ = p[2] # longitudinal dispersivity
        c_in = p[3] # inflow concentration
        # basic variables of transport
        v = q/ϕ # velocity
        De = Dₑ + αₗ*v # effective dispersion coefficient
        # transport
        c_advec = [c_in;u]
        advec = -v .* diff(c_advec) ./ Δx
        gradc = diff(u)./ Δx
        disp = ([gradc; [0]]-[[0]; gradc]).* De ./ Δx
        du .= advec .+ disp
        nothing
    end
    return ADEST!
end