import sympy as sp
import numpy as np
from sympy import exp, erfc, sqrt
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numba


"""
Creating the algebraic expression for the analytical solution of the transport equation (Lapidus and Amundson, 1952)
Using SYmpy to calculate the symbolic model and derivatives of the model.
"""
x, t, c0, c_in, phi, q, De, alpha_l, v, Dl = sp.symbols("x t c0 c_in phi q De alpha_l v Dl")
t_pulse = sp.symbols("t_pulse")
const_expr = c_in + (c0 - c_in) / 2 * erfc((x - v * t) / (2 * sqrt(Dl * t))) \
            + exp(v * x / Dl) \
            * erfc((x + v * t) / (2 * sqrt(Dl * t)))

pulse_expr = const_expr - (c0 - c_in)/ 2 * (erfc((x - v * (t - t_pulse))  \
    / (2 * sqrt(Dl * (t - t_pulse)))) \
    + exp(v * x / Dl) \
    * erfc((x + v * (t - t_pulse)) / (2 * sqrt(Dl * (t - t_pulse)))))

pulse_sympy = sp.Piecewise((const_expr, t < t_pulse), (pulse_expr, True))

pulse_sympy = pulse_sympy.subs(Dl, De + alpha_l*v)
pulse_sympy = pulse_sympy.subs(v, q/phi)
pulse_sympy

dphi = pulse_sympy.diff(phi)
dalpha_L = pulse_sympy.diff(alpha_l)

f = sp.lambdify((x, t, c0, c_in, phi, q, De, alpha_l, t_pulse), pulse_sympy, "math")
jitted_f = numba.jit(f, nopython=True)

dfdphi = sp.lambdify((x, t, c0, c_in, phi, q, De, alpha_l, t_pulse), dphi, "math")
dfdaL = sp.lambdify((x, t, c0, c_in, phi, q, De, alpha_l, t_pulse), dalpha_L, "math")
jitte_dfdphi = numba.jit(dfdphi, nopython=True)
jitte_dfdaL = numba.jit(dfdaL, nopython=True)


@numba.jit(nopython=True)
def vec_pulse(cr:np.ndarray,
              x:np.ndarray,
              t:np.ndarray,
              c0:np.float64,
              c_in:np.float64,
              phi:np.float64,
              q:np.float64,
              De:np.float64,
              alpha_l:np.float64,
              t_pulse:np.float64) -> np.ndarray:
    for i in range(x.shape[0]):
        for j in range(t.shape[0]):
            cr[j,i] = jitted_f(x[i], t[j], c0, c_in, phi, q, De, alpha_l, t_pulse)
    return None

@numba.jit(nopython=True, parallel=True)
def vec_par_pulse(cr:np.ndarray,
                  x:np.ndarray,
                  t:np.ndarray,
                  c0:np.float64,
                  c_in:np.float64,
                  phi:np.float64,
                  q:np.float64,
                  De:np.float64,
                  alpha_l:np.float64,
                  t_pulse:np.float64) -> np.ndarray:
    for i in numba.prange(x.shape[0]):
        for j in numba.prange(t.shape[0]):
            cr[j,i] = jitted_f(x[i], t[j], c0, c_in, phi, q, De, alpha_l, t_pulse)


@numba.jit(nopython=True, parallel=True)
def jac_pulse(    x:np.ndarray,
                  t:np.ndarray,
                  c0:np.float64,
                  c_in:np.float64,
                  phi:np.float64,
                  q:np.float64,
                  De:np.float64,
                  alpha_l:np.float64,
                  t_pulse:np.float64) -> np.ndarray:
    jac = np.zeros((t.shape[0], x.shape[0], 2))
    for i in numba.prange(x.shape[0]):
        for j in numba.prange(t.shape[0]):
            jac[j,i,0] = jitte_dfdphi(x[i], t[j], c0, c_in, phi, q, De, alpha_l, t_pulse)
            jac[j,i,1] = jitte_dfdaL(x[i], t[j], c0, c_in, phi, q, De, alpha_l, t_pulse)
    return jac

@numba.jit(nopython=True, parallel=True)
def sympy_loss(c_data, x, t, c0, c_in, phi, q, De, alpha_l, t_pulse):
    cr = np.zeros((t.shape[0], x.shape[0]))
    vec_par_pulse(cr, x, t, c0, c_in, phi, q, De, alpha_l, t_pulse)
    return np.sum((cr[:,0] - c_data)**2)

@numba.jit(nopython=True, parallel=True)
def sympy_jac(c_data, x, t, c0, c_in, phi, q, De, alpha_l, t_pulse):
    cr = np.zeros((t.shape[0], x.shape[0]))
    vec_par_pulse(cr, x, t, c0, c_in, phi, q, De, alpha_l, t_pulse)
    jac = jac_pulse(x, t, c0, c_in, phi, q, De, alpha_l, t_pulse)
    dtheta = np.sum(2*jac[:,0,0]*(cr[:,0] - c_data))
    dalpha_L = np.sum(2*jac[:,0,1]*(cr[:,0] - c_data))
    return np.array([dtheta, dalpha_L])



dalpha_L = pulse_sympy.diff(alpha_l)
dphi = pulse_sympy.diff(phi)

gradient = sp.lambdify((dalpha_L, dphi),(x, t, c0, c_in, phi, q, De, alpha_l, t_pulse))



if __name__ == "__main__":
    x = np.linspace(0, 0.5, 100)
    t = np.linspace(0.1, 72000, 100)
    c0 = 2
    c_in = 0.
    v = 1e-5
    alpha_l = 1e-3
    De = 1e-9
    Dl = 1e-9 + alpha_l * v
    t_pulse = 3600
    phi = 0.3
    q = 1e-5

    jitted_f(x[10], t[10], c0, c_in, phi, q, De, alpha_l, t_pulse)
    cr = np.zeros((t.shape[0], x.shape[0]))
    vec_pulse(cr, x, t, c0, c_in, phi, q, De, alpha_l, t_pulse)
    vec_par_pulse(cr, x, t, c0, c_in, phi, q, De, alpha_l, t_pulse)


    norm = Normalize(vmin=t.min(), vmax=t.max())
    colormap = cm.viridis

    fig, ax = plt.subplots()

    for ti in range(t.shape[0]):
        color = colormap(norm(t[ti]))
        ax.plot(x, cr[ti,:], color=color)
        #ax.plot(x, crp[ti, :], color=color, linestyle="--")

    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Time [s]")
    plt.xlabel("Distance x [m]")
    plt.ylabel("Normalized Concentration")
    plt.title("Concentration vs Distance for Different Times")
    plt.show()

    import sys
    sys.path.append('../benchmarks')

    data = np.loadtxt('benchmarks/data/data_a.csv', delimiter=',', skiprows=1)
    t = data[:,0]
    c = data[:,1]

    c0 = 2
    c_in = 0
    v = 1e-5
    alpha_l = 1e-3
    Dl = 1e-9 + alpha_l*v
    t_pulse = 3600
    col_length = 0.121
    x = np.array([col_length])
    c_model = np.zeros((len(t), len(x)))
    Q0_ml = 6 #ml/hr
    Q0 = Q0_ml*1e-6/3600 # m3/s
    diam = 0.037 # diameter of the column [m]
    area = np.pi*(diam/2)**2 # cross-sectional area of the column [m2]
    q = Q0/area # Darcy velocity [m/s]

    p = np.array([0.3, 1e-3])
    def loss(p):
        return sympy_loss(c, x, t, c0, c_in, p[0], q, De, p[1], t_pulse)
    
    def jac(p):
        return sympy_jac(c, x, t, c0, c_in, p[0], q, De, p[1], t_pulse)

    print(loss(p))
    print(jac_pulse(x, t, c0, c_in, p[0], q, De, p[1], t_pulse))
    print(jac(p))
