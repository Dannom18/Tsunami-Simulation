import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Domain and parameters
L = 100.0  # Extended domain length in km
Total_time = 1000.0  # Extended simulation time in seconds
g = 9.81  # Gravity acceleration in m/s^2
N_x = 100  # Increased number of grid points for better resolution
dx = L / (N_x + 2)  # Grid spacing
x = np.linspace(0, L, N_x + 2)  # Grid points

# Initial conditions (larger Gaussian disturbance for wave height)
h0 = 10.0  # Larger maximum wave height (in meters)
sigma = 15.0  # Wider disturbance (in km)
x0 = L / 2  # Location of the disturbance (midpoint of domain)
h_initial = h0 * np.exp(-((x[1:-1] - x0) ** 2) / (2 * sigma ** 2))
u_initial = np.ones(N_x)  # No initial velocity

# Boundary conditions (Sommerfeld radiation condition)
h_left = lambda t: 0.0  # No incoming waves on the left
h_right = lambda t: 0.0  # No incoming waves on the right
u_left = lambda t: 0.0  # No flow velocity at the left boundary
u_right = lambda t: 0.0  # No flow velocity at the right boundary

def central_difference(v, dx, i):
    return (v[i + 1] - v[i - 1]) / (2 * dx)

def pde_system(t, y):
    h = y[:N_x]
    u = y[N_x:]

    dh_dx = np.zeros(N_x)
    du_dx = np.zeros(N_x)
    dh_dt = np.zeros(N_x)
    du_dt = np.zeros(N_x)

    for i in range(1, N_x - 1):
        dh_dx[i] = central_difference(h, dx, i)
        du_dx[i] = central_difference(u, dx, i)

    # Differential equations for points next to the boundary
    dh_dt[0] = -1 / (2 * dx) * (h[0] * (u[1] - u_left(t)) - u[0] * (h[1] - h_left(t)))
    du_dt[0] = -1 / (2 * dx) * (g * (h[1] - h_left(t)) + u[0] * (u[1] - u_left(t)))
    dh_dt[N_x - 1] = -1 / (2 * dx) * (
        h[N_x - 1] * (u_right(t) - u[N_x - 2]) - u[N_x - 1] * (h_right(t) - h[N_x - 2])
    )
    du_dt[N_x - 1] = -1 / (2 * dx) * (
        g * (h_right(t) - h[N_x - 2]) + u[N_x - 1] * (u_right(t) - u[N_x - 2])
    )

    # Differential equations for other points
    for i in range(1, N_x - 1):
        dh_dt[i] = -h[i] * du_dx[i] - u[i] * dh_dx[i]
        du_dt[i] = -u[i] * du_dx[i] - g * dh_dx[i]

    return np.concatenate((dh_dt, du_dt))

# Solve the system
y0 = np.concatenate((h_initial, u_initial))
sol = solve_ivp(pde_system, t_span=(0, Total_time), y0=y0, t_eval=np.linspace(0, Total_time, 1000), method = "RK45")

# Retrieve results
h_results_no_boundary = sol.y[:N_x]
u_results_no_boundary = sol.y[N_x:]
t_points = sol.t

# Add boundary conditions to results
h_results_corrected = np.zeros((N_x + 2, len(t_points)))
h_results_corrected[0, :] = h_left(t_points)  # Left boundary
h_results_corrected[1:-1, :] = h_results_no_boundary  # Internal values
h_results_corrected[-1, :] = h_right(t_points)  # Right boundary

u_results_corrected = np.zeros((N_x + 2, len(t_points)))
u_results_corrected[0, :] = u_left(t_points)  # Left boundary
u_results_corrected[1:-1, :] = u_results_no_boundary  # Internal values
u_results_corrected[-1, :] = u_right(t_points)  # Right boundary

# Create a properly oriented meshgrid
X, T = np.meshgrid(x, t_points)

# Plot wave height (h) as a 3D surface
fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, T, h_results_corrected.T, cmap="viridis", edgecolor="none")
ax.set_xlabel("x (Spatial Grid)")
ax.set_ylabel("t (Time)")
ax.set_zlabel("h (Wave Height)")
ax.set_title("Large Tsunami Wave Height Over Time")
plt.show()

# Plot wave velocity (u) as a 3D surface
fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, T, u_results_corrected.T, cmap="plasma", edgecolor="none")
ax.set_xlabel("x (Spatial Grid)")
ax.set_ylabel("t (Time)")
ax.set_zlabel("u (Wave Velocity)")
ax.set_title("Large Tsunami Wave Velocity Over Time")
plt.show()


h_at_x50 = h_results_corrected[50, :]  # Wave height at x = 50 km over all time points
plt.figure(figsize=(10, 6))
plt.plot(t_points, h_at_x50)
plt.title('Wave Height at x = 50 km Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Wave Height h (m)')
plt.grid(True)
plt.show()

h_results_corrected