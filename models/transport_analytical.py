import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib import cm
from matplotlib.colors import Normalize

"""
This module contains the numpy/scipy implementation of the analytical solution of the transport equation (Lapidus and Amundson, 1952)
 for a constant and pulse injection of a solute in a 1D domain. The analytical solution is given by:
    cr(x, t) = c_in + (c0-c_in)/2*(erfc((x-v*t)/(2*sqrt(Dl*t))) + exp(v*x/Dl)*erfc((x+v*t)/(4*sqrt(Dl*t))))
"""


def constant_injection(
    cr: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    c0: np.float64,
    c_in: np.float64,
    v: np.float64,
    Dl: np.float64,
) -> np.ndarray:
    #cr = np.zeros((t.shape[0], x.shape[0]))
    for i in range(x.shape[0]):
        cr[:, i] = c_in + (c0 - c_in) / 2 * (
            scipy.special.erfc((x[i] - v * t) / (2 * np.sqrt(Dl * t)))
            + np.exp(v * x[i] / Dl)
            * scipy.special.erfc((x[i] + v * t) / (2 * np.sqrt(Dl * t)))
        )


def pulse_injection(
    cr: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    c0: np.float64,
    c_in: np.float64,
    v: np.float64,
    Dl: np.float64,
    t_pulse: np.float64,
) -> np.ndarray:
    tau = np.zeros_like(t)
    tau[t >= t_pulse] = t_pulse
    for i in range(x.shape[0]):
        cr[:, i] = c_in + (c0 - c_in) / 2 * (
            scipy.special.erfc((x[i] - v * t) / (2 * np.sqrt(Dl * t)))
            + np.exp(v * x[i] / Dl)
            * scipy.special.erfc((x[i] + v * t) / (2 * np.sqrt(Dl * t)))
        )
        co = (
            -(c0 - c_in)
            / 2
            * (
                scipy.special.erfc(
                    (x[i] - v * (t - tau)) / (2 * np.sqrt(Dl * (t - tau)))
                )
                + np.exp(v * x[i] / Dl)
                * scipy.special.erfc(
                    (x[i] + v * (t - tau)) / (2 * np.sqrt(Dl * (t - tau)))
                )
            )
        )
        cr[:, i] = cr[:, i] + co


# run the following code to test the functions
if __name__ == "__main__":
    x = np.linspace(0, 0.5, 100)
    t = np.linspace(0.1, 72000, 100)
    c0 = 1
    c_in = 0
    v = 1e-5
    alpha_l = 1e-3
    Dl = 1e-9 + alpha_l * v
    t_pulse = 3600

    cr = np.zeros((t.shape[0], x.shape[0]))
    crp = np.zeros((t.shape[0], x.shape[0]))
    constant_injection(cr, x, t, c0, c_in, v, Dl)
    pulse_injection(crp, x, t, c0, c_in, v, Dl, t_pulse)

    norm = Normalize(vmin=t.min(), vmax=t.max())
    colormap = cm.viridis

    fig, ax = plt.subplots()

    for ti in range(t.shape[0]):
        color = colormap(norm(t[ti]))
        # plt.plot(x, cr[ti,:], color=color)
        ax.plot(x, crp[ti, :], color=color, linestyle="--")

    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Time [s]")
    plt.xlabel("Distance x [m]")
    plt.ylabel("Normalized Concentration")
    plt.title("Concentration vs Distance for Different Times")
    plt.show()
