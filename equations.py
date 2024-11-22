def pde_system(t, y, dx, h_neighbors, u_neighbors):
    h, u = y
    dh_dx = (h_neighbors[1] - h_neighbors[0]) / dx
    du_dx = (u_neighbors[1] - u_neighbors[0]) / dx
    
    # PDE equations
    dh_dt = -h * du_dx - u * dh_dx
    du_dt = -u * du_dx - g * dh_dx
    return [dh_dt, du_dt]
