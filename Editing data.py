import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd


#First column is longitude (West-East) and second one is Latitude (North-South)

dataframe1 = pd.read_csv('Dataset1 - East Japan.csv', header=None, names=['Longitude', 'Latitude', 'Depth'])

grid = dataframe1.pivot(index='Latitude', columns='Longitude', values='Depth')

lon = grid.columns.values  
lat = grid.index.values    
depth_grid = grid.values   
Nx, Ny = 102, 102

lon1 = lon.copy()
lat1 = lat.copy()
depth_grid1 = depth_grid.copy()

print(len(lon))
print(len(lat))


quolon,remlon=divmod(len(lon),Nx)
quolat,remlat=divmod(len(lat),Ny)

lon = lon[:-remlon]  
lat = lat[:-remlat]  

lon_reduced = lon.reshape(-1, quolon).mean(axis=1)
lat_reduced = lat.reshape(-1, quolat).mean(axis=1)
lon_reduced=np.arange(1, Nx + 1)  # Array from 1 to Nx
lat_reduced = np.arange(1, Ny + 1)
depth_grid_reduced = depth_grid[:quolat * Ny, :quolon * Nx].reshape(quolat, Ny, quolon, Nx).mean(axis=(2, 0))

# Plot the data
plt.figure(figsize=(10, 6))
plt.contourf(lon_reduced, lat_reduced, depth_grid_reduced, cmap='viridis')  # Use contourf for a filled contour plot
plt.colorbar(label='Sea Depth')
plt.title('Sea Depth Map')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

plt.figure(figsize=(10, 6))
plt.contourf(lon1, lat1, depth_grid1, cmap='viridis')  # Use contourf for a filled contour plot
plt.colorbar(label='Sea Depth')
plt.title('Sea Depth Map')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()


print("Shape of depth_grid_reduced:", depth_grid_reduced.shape)
np.savez('reduced_data.npz', lon=lon_reduced, lat=lat_reduced, depth=depth_grid_reduced)



   #h[0, :] = h[1, :]  # Top boundary (reflective)
    #h[-1, :] = h[-2, :]  # Bottom boundary (reflective)
    #h[:, 0] = h[:, 1]  # Left boundary (reflective)
    #h[:, -1] = h[:, -2]  # Right boundary (reflective)

    #u[0, :] = -u[1, :]  # Top boundary (reflective)
    #u[-1, :] = -u[-2, :]  # Bottom boundary (reflective)
    #u[:, 0] = -u[:, 1]  # Left boundary (reflective)
    #u[:, -1] = -u[:, -2]  # Right boundary (reflective)

    #v[0, :] = -v[1, :]  # Top boundary (reflective)
    #v[-1, :] = -v[-2, :]  # Bottom boundary (reflective)
    #v[:, 0] = -v[:, 1]  # Left boundary (reflective)
    #v[:, -1] = -v[:, -2]  # Right boundary (reflective)



