import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parameters
L = 10.0
T = 2.0
g = 9.81
N_x = 10
N_t = 200
dx = L / (N_x - 1)
x = np.linspace(0, L, N_x)

# Initial conditions
h_initial = np.ones(N_x)
u_initial = np.zeros(N_x)

# Add a perturbation as for the moment initial conditions are regular
h_initial[N_x // 2] += 1.0

# Initialize arrays for h and u
h = h_initial.copy()
u = u_initial.copy()

# Define the ODE system at each spatial point
def pde_system(t, y, dx, h_neighbors, u_neighbors):

    h, u = y

    dh_dx = (h_neighbors[1] - h_neighbors[0]) / dx
    du_dx = (u_neighbors[1] - u_neighbors[0]) / dx
    
    dh_dt = -h * du_dx - u * dh_dx
    du_dt = -u * du_dx - g * dh_dx
    return [dh_dt, du_dt]

# Time-stepping
t_eval = np.linspace(0, T, N_t)
h_results = [h_initial.copy()]
u_results = [u_initial.copy()]

for t_idx in range(len(t_eval) - 1):
    
    t_span = [t_eval[t_idx], t_eval[t_idx + 1]]
    h_new = np.zeros_like(h)
    u_new = np.zeros_like(u)

    for i in range(1, N_x - 1):  # Loop over spatial grid (excluding boundaries)
        # Neighbors for finite differences
        h_neighbors = [h[i - 1], h[i + 1]]
        u_neighbors = [u[i - 1], u[i + 1]]

        # Solve ODE for this spatial point
        sol = solve_ivp(
            pde_system,
            t_span,
            [h[i], u[i]],
            args=(dx, h_neighbors, u_neighbors),
            method="RK45",
            rtol=1e-5,
            atol=1e-8,
        )
        
        # Update h and u at this point
        h_new[i], u_new[i] = sol.y[:, -1]

    # Apply boundary conditions (e.g., periodic)
    h_new[0], u_new[0] = h_new[-2], u_new[-2]
    h_new[-1], u_new[-1] = h_new[1], u_new[1]

    # Store results
    h_results.append(h_new)
    u_results.append(u_new)

    # Update h and u for the next time step
    h = h_new.copy()
    u = u_new.copy()

# Convert results to arrays
h_results = np.array(h_results)
u_results = np.array(u_results)

# Create a meshgrid for the 3D plot

X, T = np.meshgrid(x, t_eval)



# Plot height (h) as a 3D surface

fig = plt.figure(figsize=(14, 8))

ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, T, h_results, cmap='viridis', edgecolor='none')

ax.set_xlabel('x (Spatial Grid)')

ax.set_ylabel('t (Time)')

ax.set_zlabel('h (Height)')

ax.set_title('Height Profile Over Time')

plt.show()



# Plot velocity (u) as a 3D surface

fig = plt.figure(figsize=(14, 8))

ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, T, u_results, cmap='plasma', edgecolor='none')

ax.set_xlabel('x (Spatial Grid)')

ax.set_ylabel('t (Time)')

ax.set_zlabel('u (Velocity)')

ax.set_title('Velocity Profile Over Time')

plt.show()