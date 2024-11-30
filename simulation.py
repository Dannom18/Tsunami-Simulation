import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

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