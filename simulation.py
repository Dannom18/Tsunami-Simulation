import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd


#Import seadepth data
data = np.load('reduced_data.npz')
lon = data['lon']
lat = data['lat']
depth_grid = data['depth']

# Parameters
Lx, Ly = 100 * 10**3, 100 * 10**3
Total_time = 1000.0
g = 9.81
viscosity = 10.0
Nx, Ny = 100, 100
dx, dy = Lx / (Nx + 2), Ly / (Ny + 2)
x_coord = np.linspace(0, Lx, Nx + 2)
y_coord = np.linspace(0, Ly, Ny + 2)
h0 = 4000
delta_t = 0.25
ch=2.5/(Lx*Nx*100)
depth = depth_grid.reshape((Nx + 2, Ny + 2))

# Initial Conditions
exp0 = 1.0
wave_width = 10000.0
x_center, y_center = Lx / 10, Ly / 10
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
    sx = np.zeros_like(depth)
    sy = np.zeros_like(depth)

    # Finite differences for interior points
    dh_dx[1:-1, 1:-1] = (h[2:, 1:-1] - h[:-2, 1:-1]) / (2 * dx)
    dh_dy[1:-1, 1:-1] = (h[1:-1, 2:] - h[1:-1, :-2]) / (2 * dy)
    du_dx[1:-1, 1:-1] = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dx)
    du_dy[1:-1, 1:-1] = (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dy)
    dv_dx[1:-1, 1:-1] = (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dx)
    dv_dy[1:-1, 1:-1] = (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dy)
    sx[1:-1, 1:-1] = (depth[2:, 1:-1] - depth[:-2, 1:-1]) / (2 * dx)
    sy[1:-1, 1:-1] = (depth[1:-1, 2:] - depth[1:-1, :-2]) / (2 * dy)

    

    dh_dt = np.zeros_like(h)
    du_dt = np.zeros_like(u)
    dv_dt = np.zeros_like(v)

    dh_dt[1:-1, 1:-1] = -h[1:-1, 1:-1] * (du_dx[1:-1, 1:-1] + dv_dy[1:-1, 1:-1]) - u[1:-1, 1:-1] * dh_dx[1:-1, 1:-1] - v[1:-1, 1:-1] * dh_dy[1:-1, 1:-1]
    du_dt[1:-1, 1:-1] = -u[1:-1, 1:-1] * du_dx[1:-1, 1:-1] - v[1:-1, 1:-1] * du_dy[1:-1, 1:-1] - g * (dh_dx[1:-1, 1:-1])+ viscosity * ((u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx**2 + (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) / dy**2 - h[1:-1, 1:-1]*ch*g*sx[1:-1, 1:-1])
    dv_dt[1:-1, 1:-1] = -u[1:-1, 1:-1] * dv_dx[1:-1, 1:-1] - v[1:-1, 1:-1] * dv_dy[1:-1, 1:-1] - g * (dh_dy[1:-1, 1:-1]) + viscosity * ((v[2:, 1:-1] - 2 * v[1:-1, 1:-1] + v[:-2, 1:-1]) / dx**2 + (v[1:-1, 2:] - 2 * v[1:-1, 1:-1] + v[1:-1, :-2]) / dy**2 - h[1:-1, 1:-1]*ch*g*sy[1:-1, 1:-1])

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
        u_next[:, 0] = 0
        u_next[:, -1] = 0
        v_next[:, 0] = 0
        v_next[:, -1] = 0
        h_next[:, 0] = h_next[:, 1]
        h_next[:, -1] = h_next[:, -2]
        u_next[0, :] = 0
        u_next[-1, :] = 0
        v_next[0, :] = 0
        v_next[-1, :] = 0
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

# Define parameters for the seafloor slope
a, b, c = 0.05, 0.05, 0  # Slope coefficients

# Define the seafloor
seafloor = a * x_mesh + b * y_mesh + c 

def animate(i):
    ax.clear()
    data = h_data[i].T - h0
    surf = ax.plot_surface(x_mesh, y_mesh, data, cmap="Blues", rstride=4, cstride=4, linewidth=0, antialiased=False)
    #ax.contourf(x_mesh, y_mesh, depth_grid.T, cmap='viridis', alpha=0.6)
    ax.set_zlim(-max_amplitude, max_amplitude)
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Surface Elevation (m)')
    ax.set_title(f"Tsunami Wave at Time = {i * delta_t * 10:.1f} s")
    return surf,

ani = animation.FuncAnimation(fig, animate, frames=len(h_data), interval=20, blit=True)
ani.save('tsunami_wave.gif', writer='ffmpeg', fps=30)

plt.figure(figsize=(10, 6))
plt.contourf(lon, lat, depth_grid, cmap='viridis')  # Use contourf for a filled contour plot
plt.colorbar(label='Sea Depth')
plt.title('Sea Depth Map')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
