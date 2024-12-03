import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
L = 100 * 10**3.0
Total_time = 2000.0
g = 9.81
viscosity = 10.0
N_x = 2000
dx = L / (N_x + 2)
x = np.linspace(0, L, N_x + 2)
h0 = 4000

exp0 = 1.0  # Wave amplitude in meters
wave_width = 10000.0  # Wave width in meters
x_center = L/2 # Center of the wave
exp = exp0 * np.exp(-((x - x_center)**2) / (2 * wave_width**2))

h_initial = h0 + exp
u_initial = np.zeros_like(x)

delta_t = 0.25

def pde_system_viscosity(t, y):
    h = y[:N_x + 2]
    u = y[N_x + 2:]

    dh_dx = np.zeros(N_x + 2, dtype=np.float64)
    du_dx = np.zeros(N_x + 2, dtype=np.float64)
    dh_dt = np.zeros(N_x + 2, dtype=np.float64)
    du_dt = np.zeros(N_x + 2, dtype=np.float64)

    dh_dx[1:-1] = (h[2:] - h[:-2]) / (2 * dx)
    du_dx[1:-1] = (u[2:] - u[:-2]) / (2 * dx)

    dh_dt[1:-1] = -h[1:-1] * du_dx[1:-1] - u[1:-1] * dh_dx[1:-1]
    du_dt[1:-1] = -u[1:-1] * du_dx[1:-1] - g * dh_dx[1:-1] + viscosity * (u[2:] - 2 * u[1:-1] + u[:-2]) / (dx**2)

    # Apply boundary conditions
    u[0] = u[1]
    h[0] = h[1]

    u[-1] = u[-2]
    h[-1] = h[-2]

    return np.concatenate((dh_dt, du_dt))

def rungeKutta(y0: np.ndarray, tspan: tuple, delta_t: float, ode: callable):
    t0, tend = tspan
    t = [t0]
    y = [y0]

    # Determine the number of time steps
    num_steps = int((tend - t0) / delta_t)
    
    # For animation, store data at each frame
    h_data = []
    u_data = []

    for step in range(num_steps):
        k1 = delta_t * ode(t0, y[-1])
        k2 = delta_t * ode(t0 + 0.5 * delta_t, y[-1] + 0.5 * k1)
        k3 = delta_t * ode(t0 + 0.5 * delta_t, y[-1] + 0.5 * k2)
        k4 = delta_t * ode(t0 + delta_t, y[-1] + k3)

        y_next = y[-1] + (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t0 += delta_t
        t.append(t0)
        y.append(y_next)

        # Store data for animation every few steps to reduce data size
        if step % 40 == 0:
            h_current = y_next[:N_x + 2]
            u_current = y_next[N_x + 2:]
            h_data.append(h_current - h0)
            u_data.append(u_current)

    return t, y, h_data, u_data

t0 = 0.0
tf = Total_time
y0 = np.concatenate((h_initial, u_initial))

t, y, h_data, u_data = rungeKutta(y0, (t0, tf), delta_t, pde_system_viscosity)
h_data = np.array(h_data)

fig, ax = plt.subplots(figsize=(12, 6))
line, = ax.plot([], [], lw=2)
ax.set_xlim(0, L)
ax.set_ylim(-1.5, 1.5)
ax.set_xlabel('Distance (m)')
ax.set_ylabel('Surface Elevation (m)')
ax.set_title('Tsunami Wave Propagation')

def init():
    line.set_data([], [])
    return line,

def animate(i):
    x_data = x
    y_data = h_data[i]
    line.set_data(x_data, y_data)
    ax.set_title(f'Tsunami Wave at Time = {i * delta_t * 40:.1f} s')
    return line,

num_frames = h_data.shape[0]
ani = animation.FuncAnimation(fig, animate, frames=num_frames, init_func=init, blit=True, interval=50)
plt.show()