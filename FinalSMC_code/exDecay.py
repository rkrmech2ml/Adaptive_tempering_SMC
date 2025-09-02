import numpy as np
import matplotlib.pyplot as plt

# Parameters for exponential decay
max_depth = 10  # cm
initial_volume = 100  # arbitrary units
decay_rate = 0.3  # per cm

# Generate depth values
depth = np.linspace(0, max_depth, 100)
# Exponential decay formula
volume = initial_volume * np.exp(-decay_rate * depth)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(depth, volume, 'r:', label='Exponential Decay')
plt.xlabel('Depth of Non-Woven')
plt.ylabel('Deposition Volume of Droplets')
plt.title('Exponential Decay of Droplet Deposition with Depth')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
# Parameters for Gaussian distribution
mean_diameter = 5  # micrometers
std_dev = 1.5  # micrometers

# Generate droplet diameter values
diameters = np.linspace(mean_diameter - 4*std_dev, mean_diameter + 4*std_dev, 200)
# Gaussian formula
gaussian = (1/(std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((diameters - mean_diameter)/std_dev)**2)

# Plot Gaussian distribution
plt.figure(figsize=(8, 5))
plt.plot(diameters, gaussian, 'b-', label='Gaussian Distribution')
plt.axvline(mean_diameter, color='g', linestyle='--', label='Mean')
plt.axvline(mean_diameter - std_dev, color='m', linestyle=':')
plt.axvline(mean_diameter + std_dev, color='m', linestyle=':')
plt.xlabel('Droplet Diameter (μm)')
plt.ylabel('Probability Density')
plt.title('Distribution of Droplet Diameters')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()