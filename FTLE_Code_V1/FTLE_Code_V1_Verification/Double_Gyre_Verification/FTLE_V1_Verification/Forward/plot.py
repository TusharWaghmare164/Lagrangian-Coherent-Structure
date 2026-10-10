import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load CSV
data = pd.read_csv('Hallerdata.csv')

# Extract columns
x = data.iloc[:, 0].values
y = data.iloc[:, 1].values
ftle = data.iloc[:, 2].values

# Create smooth contour plot
plt.figure(figsize=(16, 8))

contour = plt.tricontourf(
    x, y, ftle,
    levels=1024,      # More levels = smoother
    cmap='jet'
)

# Colorbar
plt.colorbar(contour, label='FTLE')

# Labels
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(0,2)
plt.ylim(0,1)

# Equal scaling
plt.axis('equal')

plt.tight_layout()
plt.show()