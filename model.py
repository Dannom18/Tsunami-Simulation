import numpy as np


def pde_system(t, y, N_x, dx, g = 9.81):
    h = y[:N_x]
    u = y[N_x:]

    # Compute spatial derivatives
    dh_dx = np.zeros(N_x)
    du_dx = np.zeros(N_x)

    # Interior points (central differences)
    dh_dx[1:-1] = (h[2:] - h[:-2]) / (2 * dx)
    du_dx[1:-1] = (u[2:] - u[:-2]) / (2 * dx)

    # Boundary points (one-sided differences)
    dh_dx[0] = (h[1] - h[0]) / dx
    du_dx[0] = (u[1] - u[0]) / dx
    dh_dx[-1] = (h[-1] - h[-2]) / dx
    du_dx[-1] = (u[-1] - u[-2]) / dx

    # Time derivatives
    dh_dt = -h * du_dx - u * dh_dx
    du_dt = -u * du_dx - g * dh_dx

    return np.concatenate((dh_dt, du_dt))