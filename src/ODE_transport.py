import numpy as np
from numba import jit

@jit(nopython=True)
def ADEST(du, u, p, t, c_in, De, q, dx):
    """
    Calculate the right-hand side of the transport equation.

    Parameters:
    - du: The derivative of u with respect to time.
    - u: The concentration array.
    - p: Parameters array where p[0] is porosity (ϕ) and p[1] is longitudinal dispersivity (αₗ).
    - t: Time variable.
    - c_in: The inflow concentration.
    - De: The effective diffusion coefficient.
    - q: The specific flow rate.
    - dx: The spatial step size.

    Returns:
    - Updates du in place with the right-hand side of the transport equation.
    """
    ϕ = p[0]  # porosity
    αₗ = p[1]  # longitudinal dispersivity

    # basic variables of transport
    v = q / ϕ  # velocity
    De_eff = De + αₗ * v  # effective dispersion coefficient

    # transport
    c_advec = np.concatenate(([c_in], u))
    advec = -v * np.diff(c_advec) / dx
    gradc = np.diff(u) / dx
    disp = (np.concatenate((gradc, [0])) - np.concatenate(([0], gradc))) * De_eff / dx
    du[:] = advec + disp

# Example usage
du = np.zeros(10)
u = np.random.rand(10)
p = np.array([0.3, 1e-3])
t = 0
c_in = 1.0
De = 1e-9
q = 1e-5
dx = 0.1

ADEST(du, u, p, t, c_in, De, q, dx)
print(du)