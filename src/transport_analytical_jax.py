import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from matplotlib import cm
from matplotlib.colors import Normalize

"""
This module contains the jax.numpy/jax.scipy implementation of the analytical solution of the transport equation (Lapidus and Amundson, 1952)
 for a constant and pulse injection of a solute in a 1D domain. The analytical solution is given by:
    cr(x, t) = c_in + (c0-c_in)/2*(erfc((x-v*t)/(2*sqrt(Dl*t))) + exp(v*x/Dl)*erfc((x+v*t)/(4*sqrt(Dl*t))))
"""


@jax.jit
def constant_injection(
    x: jnp.ndarray,
    t: jnp.ndarray,
    c0: jnp.float64,
    c_in: jnp.float64,
    v: jnp.float64,
    Dl: jnp.float64,
) -> jnp.ndarray:
    def compute_cr(i):
        return c_in + (c0 - c_in) / 2 * (
            jscipy.special.erfc((x[i] - v * t) / (2 * jnp.sqrt(Dl * t)))
            + jnp.exp(v * x[i] / Dl)
            * jscipy.special.erfc((x[i] + v * t) / (2 * jnp.sqrt(Dl * t)))
        )

    cr = jax.vmap(compute_cr)(jnp.arange(x.shape[0]))
    return cr.T


@jax.jit
def pulse_injection(
    x: jnp.ndarray,
    t: jnp.ndarray,
    c0: jnp.float64,
    c_in: jnp.float64,
    v: jnp.float64,
    Dl: jnp.float64,
    t_pulse: jnp.float64,
) -> jnp.ndarray:
    tau = jnp.where(t >= t_pulse, t_pulse, 0)

    def compute_cr(i):
        c_init = c_in + (c0 - c_in) / 2 * (
            jscipy.special.erfc((x[i] - v * t) / (2 * jnp.sqrt(Dl * t)))
            + jnp.exp(v * x[i] / Dl)
            * jscipy.special.erfc((x[i] + v * t) / (2 * jnp.sqrt(Dl * t)))
        )
        co = (
            -(c0 - c_in)
            / 2
            * (
                jscipy.special.erfc(
                    (x[i] - v * (t - tau)) / (2 * jnp.sqrt(Dl * (t - tau)))
                )
                + jnp.exp(v * x[i] / Dl)
                * jscipy.special.erfc(
                    (x[i] + v * (t - tau)) / (2 * jnp.sqrt(Dl * (t - tau)))
                )
            )
        )
        return c_init + co

    cr = jax.vmap(compute_cr)(jnp.arange(x.shape[0]))
    return cr.T


# run the following code to test the functions
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x = jnp.linspace(0, 0.5, 100)
    t = jnp.linspace(0.1, 72000, 100)
    c0 = 1
    c_in = 0
    v = 1e-5
    alpha_l = 1e-3
    Dl = 1e-9 + alpha_l * v
    t_pulse = 3600

    cr = jnp.zeros((t.shape[0], x.shape[0]))

    cr = constant_injection(x, t, c0, c_in, v, Dl)
    cr_pulse = pulse_injection(x, t, c0, c_in, v, Dl, t_pulse)

    norm = Normalize(vmin=t.min(), vmax=t.max())
    colormap = cm.viridis

    fig, ax = plt.subplots()

    for ti in range(t.shape[0]):
        color = colormap(norm(t[ti]))
        ax.plot(x, cr[ti, :], color=color)
        ax.plot(x, cr_pulse[ti, :], color=color, linestyle="--")

    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Time [s]")
    plt.xlabel("Distance x [m]")
    plt.ylabel("Normalized Concentration")
    plt.title("Concentration vs Distance for Different Times")
    plt.show()
