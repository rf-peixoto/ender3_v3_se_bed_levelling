import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import griddata

# Measured Z error values at each point on the bed (in micrometers)
data_micrometers = np.array([
    [-0.04, -0.25, -0.33, -0.33],
    [ 0.03, -0.12, -0.20, -0.28],
    [ 0.29,  0.17,  0.04, -0.03],
    [ 0.80,  0.64,  0.53,  0.47]
])

# Convert Z-error values from micrometers to millimeters for plotting
data = data_micrometers / 1000.0  # 1 mm = 1000 µm

# Flip the data if necessary to match the bed's orientation
# data = np.flipud(data)  # Uncomment if the orientation is incorrect

# Define the bed dimensions (in millimeters)
bed_size = 220  # Bed size in mm
num_divisions = data.shape[0]  # Number of divisions along one axis (4 in this case)
region_size = bed_size / num_divisions  # Size of each region

# Generate X and Y positions at the centers of each region
x_positions = np.array([(i + 0.5) * region_size for i in range(num_divisions)])
y_positions = np.array([(i + 0.5) * region_size for i in range(num_divisions)])
X_measure, Y_measure = np.meshgrid(x_positions, y_positions)

# Flatten the measurement arrays for interpolation
points = np.vstack((X_measure.flatten(), Y_measure.flatten())).T
values = data.flatten()

# Create a grid over the entire bed for plotting
grid_resolution = 100  # Adjust for higher or lower resolution
grid_x, grid_y = np.meshgrid(
    np.linspace(0, bed_size, grid_resolution),
    np.linspace(0, bed_size, grid_resolution)
)

# Interpolate the Z-error values over the grid
grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')

# Create a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(
    grid_x, grid_y, grid_z,
    cmap=cm.viridis,
    edgecolor='none',
    antialiased=True,
    alpha=0.8  # Slight transparency to see markers better
)

# Plot the measurement points
scatter = ax.scatter(
    X_measure.flatten(),
    Y_measure.flatten(),
    data.flatten(),
    c=data.flatten(),
    cmap=cm.coolwarm,
    edgecolor='k',
    s=50,
    marker='o'
)

# Add a color bar for the measurement points
cbar_scatter = fig.colorbar(scatter, shrink=0.5, aspect=10, pad=0.05)
cbar_scatter.set_label('Measurement Z Error (mm)')

# Add a color bar for the surface plot
cbar_surface = fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.1)
cbar_surface.set_label('Interpolated Z Error (mm)')

# Set labels and title
ax.set_xlabel('X Position (mm)')
ax.set_ylabel('Y Position (mm)')
ax.set_zlabel('Z Error (mm)')
ax.set_title('3D Printer Bed Leveling Visualization')

# Set axis limits
ax.set_xlim(0, bed_size)
ax.set_ylim(0, bed_size)

# Adjust the Z-axis limits to emphasize the small Z-errors
max_z_error = np.max(np.abs(data))
ax.set_zlim(-max_z_error * 1.5, max_z_error * 1.5)

# Adjust the aspect ratio using set_box_aspect (requires Matplotlib 3.3+)
ax.set_box_aspect((bed_size, bed_size, max_z_error * 1000))  # Scale Z to match X and Y

# Adjust the view angle for better visualization (optional)
ax.view_init(elev=30, azim=225)

# Show the plot
plt.tight_layout()
plt.show()
