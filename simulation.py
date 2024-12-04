import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# Parameters
Lx, Ly = 100 * 10**3, 100 * 10**3
Total_time = 10000.0
g = 9.81
viscosity = 10.0
Nx, Ny = 100, 100
dx, dy = Lx / (Nx + 2), Ly / (Ny + 2)
x_coord = np.linspace(0, Lx, Nx + 2)
y_coord = np.linspace(0, Ly, Ny + 2)
h0 = 4000
delta_t = 0.25

# Initial Conditions
exp0 = 1.0
wave_width = 10000.0
x_center, y_center = Lx / 2, Ly / 2
exp = exp0 * np.exp(-((x_coord[:, None] - x_center)**2 + (y_coord[None, :] - y_center)**2) / (2 * wave_width**2))

h_initial = h0 + exp
u_initial = np.zeros((Nx + 2, Ny + 2))
v_initial = np.zeros((Nx + 2, Ny + 2))

delta_t = 0.25

def pde_system_2D(t, y):
    h = y[: (Nx + 2) * (Ny + 2)].reshape((Nx + 2, Ny + 2))
    u = y[(Nx + 2) * (Ny + 2) : 2 * (Nx + 2) * (Ny + 2)].reshape((Nx + 2, Ny + 2))
    v = y[2 * (Nx + 2) * (Ny + 2) :].reshape((Nx + 2, Ny + 2))

    # Compute derivatives
    dh_dx = np.zeros_like(h)
    dh_dy = np.zeros_like(h)
    du_dx = np.zeros_like(u)
    du_dy = np.zeros_like(u)
    dv_dx = np.zeros_like(v)
    dv_dy = np.zeros_like(v)

    # Finite differences for interior points
    dh_dx[1:-1, 1:-1] = (h[2:, 1:-1] - h[:-2, 1:-1]) / (2 * dx)
    dh_dy[1:-1, 1:-1] = (h[1:-1, 2:] - h[1:-1, :-2]) / (2 * dy)
    du_dx[1:-1, 1:-1] = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dx)
    du_dy[1:-1, 1:-1] = (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dy)
    dv_dx[1:-1, 1:-1] = (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dx)
    dv_dy[1:-1, 1:-1] = (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dy)

    dh_dt = np.zeros_like(h)
    du_dt = np.zeros_like(u)
    dv_dt = np.zeros_like(v)

    dh_dt[1:-1, 1:-1] = -h[1:-1, 1:-1] * (du_dx[1:-1, 1:-1] + dv_dy[1:-1, 1:-1]) - u[1:-1, 1:-1] * dh_dx[1:-1, 1:-1] - v[1:-1, 1:-1] * dh_dy[1:-1, 1:-1]
    du_dt[1:-1, 1:-1] = -u[1:-1, 1:-1] * du_dx[1:-1, 1:-1] - v[1:-1, 1:-1] * du_dy[1:-1, 1:-1] - g * dh_dx[1:-1, 1:-1] + viscosity * ((u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx**2 + (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) / dy**2)
    dv_dt[1:-1, 1:-1] = -u[1:-1, 1:-1] * dv_dx[1:-1, 1:-1] - v[1:-1, 1:-1] * dv_dy[1:-1, 1:-1] - g * dh_dy[1:-1, 1:-1] + viscosity * ((v[2:, 1:-1] - 2 * v[1:-1, 1:-1] + v[:-2, 1:-1]) / dx**2 + (v[1:-1, 2:] - 2 * v[1:-1, 1:-1] + v[1:-1, :-2]) / dy**2)

    return np.concatenate((dh_dt.ravel(), du_dt.ravel(), dv_dt.ravel()))

def rungeKutta_2D(y0, tspan, delta_t, ode):
    t0, tend = tspan
    t = [t0]
    y = [y0]

    # Determine the number of time steps
    num_steps = int((tend - t0) / delta_t)
    h_data = []

    for step in range(num_steps):

        k1 = delta_t * ode(t0, y[-1])
        k2 = delta_t * ode(t0 + 0.5 * delta_t, y[-1] + 0.5 * k1)
        k3 = delta_t * ode(t0 + 0.5 * delta_t, y[-1] + 0.5 * k2)
        k4 = delta_t * ode(t0 + delta_t, y[-1] + k3)

        y_next = y[-1] + (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Apply boundary conditions
        h_next = y_next[: (Nx + 2) * (Ny + 2)].reshape((Nx + 2, Ny + 2))
        u_next = y_next[(Nx + 2) * (Ny + 2) : 2 * (Nx + 2) * (Ny + 2)].reshape((Nx + 2, Ny + 2))
        v_next = y_next[2 * (Nx + 2) * (Ny + 2) :].reshape((Nx + 2, Ny + 2))

        # Boundarie Conditions
        u_next[:, 0] = u_next[:, 1]
        u_next[:, -1] = u_next[:, -2]
        v_next[:, 0] = v_next[:, 1]
        v_next[:, -1] = v_next[:, -2]
        h_next[:, 0] = h_next[:, 1]
        h_next[:, -1] = h_next[:, -2]
        u_next[0, :] = u_next[1, :]
        u_next[-1, :] = u_next[-2, :]
        v_next[0, :] = v_next[1, :]
        v_next[-1, :] = v_next[-2, :]
        h_next[0, :] = h_next[1, :]
        h_next[-1, :] = h_next[-2, :]

        # Flatten y_next back into a vector
        y_next = np.concatenate((h_next.ravel(), u_next.ravel(), v_next.ravel()))

        t0 += delta_t
        t.append(t0)
        y.append(y_next)

        if step % 10 == 0:
            h_data.append(h_next.copy())

    return t, y, np.array(h_data)

# Solve system
y0 = np.concatenate((h_initial.ravel(), u_initial.ravel(), v_initial.ravel()))
t_values, solution_values, h_data = rungeKutta_2D(y0, (0.0, Total_time), delta_t, pde_system_2D)

# Animation
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
x_mesh, y_mesh = np.meshgrid(x_coord / 1000.0, y_coord / 1000.0)
max_amplitude = np.max(np.abs(h_data - h0))

def animate(i):
    ax.clear()
    data = h_data[i].T - h0
    surf = ax.plot_surface(x_mesh, y_mesh, data, cmap="Blues", rstride=4, cstride=4, linewidth=0, antialiased=False)
    ax.set_zlim(-max_amplitude, max_amplitude)
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Surface Elevation (m)')
    ax.set_title(f"Tsunami Wave at Time = {i * delta_t * 10:.1f} s")
    return surf,

ani = animation.FuncAnimation(fig, animate, frames=len(h_data), interval=20, blit=True)
ani.save('tsunami_wave.gif', writer='ffmpeg', fps=30)

# Domain and parameters
L = 100e3  # Extended domain length in meters
Total_time = 3000.0  # Simulation time in seconds
g = 9.81  # Gravity acceleration in m/s^2
N_x = 200  # Increased number of grid points for better resolution
dx = L / (N_x + 2)  # Grid spacing
x = np.linspace(0, L, N_x + 2)  # Grid points

# Initial conditions
h0 = 10.0  # Maximum wave height (m)
sigma = 15e3  # Disturbance width (m)
x0 = L / 2  # Disturbance center
h_initial = h0 * np.exp(-((x[1:-1] - x0) ** 2) / (2 * sigma ** 2))
u_initial = np.zeros(N_x)  # No initial velocity

def pde_system(t, y):
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

# Initial conditions
y0 = np.concatenate((h_initial, u_initial))

# Estimate maximum wave speed and time step
c_max = np.sqrt(g * (h0 + np.max(h_initial)))
dt_max = dx / c_max
num_time_points = 5000
t_eval = np.linspace(0, Total_time, num_time_points)


# Solve the system
sol = solve_ivp(
    pde_system,
    t_span=(0, Total_time),
    y0=y0,
    method="BDF",
    t_eval=t_eval,
    max_step=dt_max,
    rtol=1e-6,
    atol=1e-8
)

# Retrieve results
h_results_no_boundary = sol.y[:N_x]
u_results_no_boundary = sol.y[N_x:]
t_points = sol.t

# Add boundary conditions to results
h_results_corrected = np.zeros((N_x + 2, len(t_points)))
h_results_corrected[1:-1, :] = h_results_no_boundary
h_results_corrected[0, :] = h_results_no_boundary[0, :]
h_results_corrected[-1, :] = h_results_no_boundary[-1, :]

u_results_corrected = np.zeros((N_x + 2, len(t_points)))
u_results_corrected[1:-1, :] = u_results_no_boundary
u_results_corrected[0, :] = u_results_no_boundary[0, :]
u_results_corrected[-1, :] = u_results_no_boundary[-1, :]

# Create meshgrid for plotting
T, X = np.meshgrid(t_points, x)

# Plot wave height (h) as a 3D surface
fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(X, T, h_results_corrected, cmap="viridis", edgecolor="none")
ax.set_xlabel("x (m)")
ax.set_ylabel("Time (s)")
ax.set_zlabel("h (m)")
ax.set_title("Large Tsunami Wave Height Over Time")
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

# Plot wave velocity (u) as a 3D surface
fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(X, T, u_results_corrected, cmap="plasma", edgecolor="none")
ax.set_xlabel("x (m)")
ax.set_ylabel("Time (s)")
ax.set_zlabel("u (m/s)")
ax.set_title("Large Tsunami Wave Velocity Over Time")
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

# Plot wave height at x = L/2 over time
mid_index = N_x // 2 + 1  # Adjust for added boundary points
h_at_x_mid = h_results_corrected[mid_index, :]
plt.figure(figsize=(10, 6))
plt.plot(t_points, h_at_x_mid)
plt.title('Wave Height at x = L/2 Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Wave Height h (m)')
plt.grid(True)
plt.show()

