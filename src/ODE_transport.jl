
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
function create_ADEST(Δx, cᵢ, Dₑ, q)
    function ADEST!(du, u, p ,t)
        ϕ = p[1] # porosity
        αₗ = p[2] # longitudinal dispersivity

        # basic variables of transport
        v = q/ϕ # velocity
        De = Dₑ + αₗ*v # effective dispersion coefficient
        # transport
        c_advec = [cᵢ;u]
        advec = -v .* diff(c_advec, dims=1) ./ Δx
        gradc = diff(u, dims=1)./ Δx
        dims = size(u)
        disp = ([gradc; zeros(1, dims[2])]-[zeros(1, dims[2]); gradc]).* De ./ Δx
        du .= advec .+ disp
        nothing
    end
    return ADEST!
end