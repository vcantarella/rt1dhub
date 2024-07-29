# erfc_cython.pyx
from libc.math cimport erfc

cdef api cython_erfc(double x):
    return erfc(x)

# Expose the function
__pyx_capi__ = {
    "cython_erfc": cython_erfc
}