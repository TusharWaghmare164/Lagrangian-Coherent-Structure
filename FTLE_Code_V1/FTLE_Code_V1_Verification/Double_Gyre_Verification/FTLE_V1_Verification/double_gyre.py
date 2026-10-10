import numpy as np
import pandas as pd

# =========================
# Parameters
# =========================

A = 0.1
e = 0.1
omega = np.pi / 10

# =========================
# Domain limits
# =========================

x_range = (0, 2)
y_range = (0, 1)

dx = 0.01
dy = 0.01

# =========================
# Double gyre functions
# =========================

def f(x, t):
    T1 = e * np.sin(omega * t) * x**2
    T2 = (1 - 2 * e * np.sin(omega * t)) * x
    return T1 + T2

def dfdx(x, t):
    T1 = 2 * e * np.sin(omega * t) * x
    T2 = (1 - 2 * e * np.sin(omega * t))
    return T1 + T2

def u(x, y, t):
    T1 = np.pi * f(x, t)
    T2 = np.pi * y
    return -np.pi * A * np.sin(T1) * np.cos(T2)

def v(x, y, t):
    T1 = np.pi * f(x, t)
    T2 = np.pi * y
    return np.pi * A * np.sin(T2) * np.cos(T1) * dfdx(x, t)

# =========================
# Create mesh
# =========================

x_vals = np.arange(x_range[0], x_range[1] + dx, dx)
y_vals = np.arange(y_range[0], y_range[1] + dy, dy)

# x_vals = np.linspace(x_range[0], x_range[1], 500)
# y_vals = np.linspace(y_range[0], y_range[1], 251)

X, Y = np.meshgrid(x_vals, y_vals)

# =========================
# Time steps
# =========================

T = 2 * np.pi / omega

time_steps = np.linspace(0, T, 6)

# =========================
# Generate datasets
# =========================

for i, t in enumerate(time_steps):

    U = u(X, Y, t)
    V = v(X, Y, t)

    df = pd.DataFrame({
        'x': X.ravel(),
        'y': Y.ravel(),
        'u': U.ravel(),
        'v': V.ravel()
    })

    filename = f"double_gyre_rev_t{i}.csv"

    df.to_csv(filename, index=False)

    print(f"Saved: {filename}")