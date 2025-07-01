'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PART A ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Define parameters
grid_size = 100
voltage_precision = 1*10**(-6)
V_plus = 1.0
V_minus = -1.0

# Initialize potential grid
# Boundary conditions: Walls of box already at 0V
V = np.zeros((grid_size, grid_size))

# Box size: 10 cm x 10 cm
# Grid size: 100 x 100
# Grid cell: 0.1 cm
dx = 10/grid_size
dy = 10/grid_size

# Define positions for plates
plate_width = 6
plate_start_y = 20      # Plate starts 2 cm from bottom => 20th row
plate_end_y = 80        # Plate ends 8 cm from bottom => 80th row
plate_plus_x = 20   # + Plate 2 cm from left => 20th column
plate_minus_x = 80   # - Plate 8 cm from left => 80th column

# Set potential on plates
V[plate_start_y:plate_end_y, plate_plus_x] = V_plus
V[plate_start_y:plate_end_y, plate_minus_x] = V_minus

# Define function to perform Gauss-Seidel without over-relaxation
def gauss_seidel(V, voltage_precision):

    converged = False
    iterations = 0

    while not converged:
        max_diff = 0
        for i in range(1, grid_size-1):
            for j in range(1, grid_size-1):
                # Skip plate points
                if (j == plate_plus_x and plate_start_y <= i < plate_end_y) or (
                    j == plate_minus_x and plate_start_y <= i < plate_end_y):
                    continue

                # Update potential
                new_V = 0.25 * (V[i+1, j] + V[i-1, j] + V[i, j+1] + V[i, j-1])
                max_diff = max(max_diff, abs(new_V - V[i, j]))
                V[i, j] = new_V

        iterations += 1
        converged = max_diff < voltage_precision
    return V, iterations

# Solve using Gauss-Seidel method without over-relaxation
V_gs, gs_iterations = gauss_seidel(V.copy(), voltage_precision)

my_gradient = LinearSegmentedColormap.from_list('Random gradient 9429', (
    # Edit this gradient at https://eltos.github.io/gradient/#Random%20gradient%209429=1E33B0-8BAEFD-F0F0F0-EA6080-B3026B
    (0.000, (0.118, 0.200, 0.690)),
    (0.250, (0.545, 0.682, 0.992)),
    (0.500, (0.921, 0.921, 0.921)),
    (0.750, (0.918, 0.376, 0.502)),
    (1.000, (0.702, 0.008, 0.420))))

# Create Contour plot of results
plt.figure(figsize=(8,8))
contour = plt.contourf(V_gs, levels=50, cmap=my_gradient)
cbar = plt.colorbar(contour, label='Potential (V)')
cbar.ax.yaxis.label.set_size(15) 
plt.title(f"Electrostatic Potential (Gauss-Seidel, {gs_iterations} Iterations)", fontsize=20)
plt.xlabel("x [grid points]", fontsize=15)
plt.ylabel("y [grid points]", fontsize=15)
plt.axis('square')
plt.show()

'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PART B ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
# Define a function to perform Gauss-Seidel method with over-relaxation
def gauss_seidel_or(V, voltage_precision, omega):

    converged = False
    iterations = 0

    while not converged:
        max_diff = 0
        for i in range(1, grid_size-1):
            for j in range(1, grid_size-1):
                # Skip plate points
                if (j == plate_plus_x and plate_start_y <= i < plate_end_y) or (
                    j == plate_minus_x and plate_start_y <= i < plate_end_y):
                    continue

                # Update potential with over-relaxation
                new_V = 0.25 * (V[i+1, j] + V[i-1, j] + V[i, j+1] + V[i, j-1])
                V[i, j] = V[i, j] + omega*(new_V - V[i, j])
                max_diff = max(max_diff, abs(new_V - V[i, j]))

        iterations += 1
        converged = max_diff < voltage_precision
    return V, iterations

# Solve with omega = 0.1, omega = 0.5, omega = 1.1
for omega in [0.1, 0.5, 1.1]:
    V_or, or_iterations = gauss_seidel_or(V.copy(), voltage_precision, omega)
    plt.figure(figsize=(8, 8))
    contour = plt.contourf(V_or, levels=50, cmap=my_gradient)
    cbar = plt.colorbar(contour, label='Potential (V)')
    cbar.ax.yaxis.label.set_size(15) 
    plt.title(fr"Electrostatic Potential (Over-relaxation, $\omega$={omega}, {or_iterations} Iterations)", fontsize=20)
    plt.xlabel("x [grid points]", fontsize=15)
    plt.ylabel("y [grid points]", fontsize=15)
    plt.axis('square')
    plt.show()

'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PART C ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
# Calculate electric field components from potential
Ex, Ey = np.gradient(-V_or, dx, dy)

# Create stream plot of electric field lines
x = np.linspace(0, grid_size, grid_size)
y = np.linspace(0, grid_size, grid_size)
X, Y = np.meshgrid(x, y)

# Create stream plot of electric field lines
plt.figure(figsize=(8, 8))
plt.streamplot(X, Y, Ex, Ey, color=V_or, cmap=my_gradient)
plt.colorbar(label="Potential (V)")
plt.title(fr"Electric Field Lines ($\omega$ = {omega})", fontsize=20)
plt.xlabel("x [grid points]", fontsize=15)
plt.ylabel("y [grid points]", fontsize=15)
plt.axis('square')
plt.show()
