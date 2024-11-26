import numba
import numpy as np


def create_ADEST(dx, c_in, De, q):
    """
    Create a right hand side (rhs) function for transport given the initial conditions.
    The function is used to solve the transport equation using the DifferentialEquations.jl package.

    # Arguments
    - `dx::Real`: The spatial step size.
    - `c_in::Real`: The inflow concentration.
    - `De::Real`: The effective difusion coefficient.
    - `q::Real`: the specific flow rate.

    # Returns
        A function that calculates the right hand side of the transport equation.

    # Example
    ```julia
    ```
    """
    def ADEST(du, u, p ,t):
        ϕ = p[1] # porosity
        αₗ = p[2] # longitudinal dispersivity

        # basic variables of transport
        v = q/ϕ # velocity
        De = De + αₗ*v # effective dispersion coefficient
        # transport
        c_advec = [c_in;u]
        advec = -v .* diff(c_advec, dims=1) ./ dx
        gradc = diff(u, dims=1)./ dx
        dims = size(u)
        disp = ([gradc; zeros(1, dims[2])]-[zeros(1, dims[2]); gradc]).* De ./ dx
        du .= advec .+ disp
        nothing
    return ADEST