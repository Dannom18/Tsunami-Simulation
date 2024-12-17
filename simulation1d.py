import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters for the normal distribution
mu = 0      # mean
sigma = 1   # standard deviation

# Parameters
L = 100 * 10**3.0
Total_time = 300000.0
g = 9.81
viscosity = 10**2
Nx = 500  # number of points in the spatial domain
dx = L / Nx
x = np.linspace(0, L, Nx)

# Generate a topographic profile
x_dist = np.linspace(mu - 4*sigma, mu + 4*sigma, Nx)
cdf_values = norm.cdf(x_dist, loc=mu, scale=sigma)
coast = cdf_values  # This represents bed elevation (B)

u_left = lambda t: 1
u_right = lambda t: 0

# Initial conditions: h is water depth above bed, not absolute water surface
# Start with a flat 0.5 m layer, then add a perturbation
h_initial = np.ones(Nx, dtype=np.float64) * 0.5
u_initial = np.zeros(Nx, dtype=np.float64)

# Add a perturbation
h_initial[0:100] += 1.0
u_initial[0:100] += 2.0

# Make sure no negative depth
h_initial = np.maximum(h_initial, 0.0)

def pde_system_vector_op(t, y):
    h = y[:Nx]
    u = y[Nx:]

    dh_dx = np.zeros(Nx, dtype=np.float64)
    du_dx = np.zeros(Nx, dtype=np.float64)
    dh_dt = np.zeros(Nx, dtype=np.float64)
    du_dt = np.zeros(Nx, dtype=np.float64)

    # Compute spatial derivatives
    dh_dx[1:-1] = (h[2:] - h[:-2]) / (2*dx)
    dh_dx[0] = (h[1] - h[0]) / dx
    dh_dx[-1] = (h[-1] - h[-2]) / dx

    du_dx[1:-1] = (u[2:] - u[:-2]) / (2*dx)
    du_dx[0] = (u[1] - u[0]) / dx
    du_dx[-1] = (u[-1] - u[-2]) / dx

    # Bed slope
    sx = np.zeros(Nx, dtype=np.float64)
    sx[1:-1] = (coast[2:] - coast[:-2]) / (2*dx)
    sx[0] = (coast[1] - coast[0]) / dx
    sx[-1] = (coast[-1] - coast[-2]) / dx

    # Shallow water equations in non-conservative form:
    # dh/dt = - (hu)_x = - (h u_x + u h_x)
    dh_dt = -(h * du_dx + u * dh_dx)

    # u_t + u u_x + g h_x = - g B_x + viscosity * u_xx/h?
    # For simplicity, just add viscosity on u separately as done before.
    u_xx = np.zeros(Nx, dtype=np.float64)
    u_xx[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / dx**2

    du_dt = (-u * du_dx - g * dh_dx - g * sx + viscosity * u_xx)

    # Apply reflective BCs on du_dt and dh_dt (optional)
    # Here we keep them as is since we handle them in time stepper

    return np.concatenate((dh_dt, du_dt))

def compute_time_step_1D(y):
    safety_factor = 0.7
    h = y[:Nx]
    u = y[Nx:]
    c = np.sqrt(g * h)
    evals_1 = np.abs(u - c)
    evals_2 = np.abs(u + c)
    max_eval = max(np.max(evals_1), np.max(evals_2), 1e-6)
    time_step = safety_factor * dx / max_eval
    return time_step

def rungeKutta_1D(y0, tspan, ode):
    t0, tend = tspan
    t = [t0]
    y = [y0.copy()]

    step = 0
    h_data = []

    while t0 < tend:
        delta_t = compute_time_step_1D(y[-1])
        if t0 + delta_t > tend:
            delta_t = tend - t0

        k1 = ode(t0, y[-1])
        k2 = ode(t0 + 0.5*delta_t, y[-1] + 0.5*delta_t*k1)
        k3 = ode(t0 + 0.5*delta_t, y[-1] + 0.5*delta_t*k2)
        k4 = ode(t0 + delta_t, y[-1] + delta_t*k3)

        y_next = y[-1] + (delta_t/6.0)*(k1 + 2*k2 + 2*k3 + k4)

        # Extract h and u
        h_next = y_next[:Nx]
        u_next = y_next[Nx:]

        # Boundary conditions: reflective
        u_next[0] = u_left(t0)
        u_next[-1] = u_right(t0)
        h_next[0] = h_next[1]
        h_next[-1] = h_next[-2]

        # Ensure no negative depth
        h_next = np.maximum(h_next, 0)

        # Recombine
        y_next = np.concatenate((h_next, u_next))
        
        t0 += delta_t
        t.append(t0)
        y.append(y_next.copy())

        step += 1
        if step % 10 == 0:
            h_data.append(h_next.copy())

    return t, y, np.array(h_data)

y0 = np.concatenate((h_initial, u_initial))
t_values, solution_values, h_data = rungeKutta_1D(y0, (0.0, Total_time), pde_system_vector_op)

solution_values = np.array(solution_values)
h_results = h_data
time_steps = len(h_results)

fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
ax.set_xlim(x.min(), x.max())
ax.set_ylim(-1, 3)
ax.set_xlabel("Distance")
ax.set_ylabel("h (Water Depth)")
ax.set_title("Time Evolution of h (Wave Runup)")
ax.plot(x, coast, label='Bed Elevation (Coast)', color='k', linestyle='--')
ax.legend()

values = h_results

def init():
    line.set_data([], [])
    return line,

def update(frame):
    y = values[frame, :]
    line.set_data(x, y)
    return line,

ani = FuncAnimation(fig, update, frames=time_steps, init_func=init, blit=True, interval=20)
plt.show()
ani.save('wave_to_coast.gif', writer='ffmpeg', fps=30)

