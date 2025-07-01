'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~PART A~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
import numpy as np

# Set constants
c = 3e8     # Speed of light in m/s
m = 1       # Mass in kg
k = 12      # Spring constant in N/m
x0 = 0.01   # Initial position in meters

# Define velocity function g(x)
def g(x, initial_position):
    term_1 = k*(initial_position**2 - x**2)
    term_2 = 2*m*c**2 + 0.5*term_1
    term_3 = (2*(m*c**2 + 0.5*term_1)**2)
    return c * np.sqrt(term_1 * term_2 / term_3)

# Use gaussxw.py to find sample points and weights
from pylab import *
def gaussxw(N):

    # Initial approximation to roots of the Legendre polynomial
    a = linspace(3,4*N-1,N)/(4*N+2)
    x = np.cos(np.pi*a+1/(8*N*N*np.tan(a)))

    # Find roots using Newton's method
    epsilon = 1e-15
    delta = 1.0
    while delta>epsilon:
        p0 = ones(N,float)
        p1 = copy(x)
        for k in range(1,N):
            p0,p1 = p1,((2*k+1)*x*p1-k*p0)/(k+1)
        dp = (N+1)*(p0-x*p1)/(1-x*x)
        dx = p1/dp
        x -= dx
        delta = np.max(np.abs(dx))

    # Calculate the weights
    w = 2*(N+1)*(N+1)/(N*N*(1-x*x)*dp*dp)

    # Return width of sample points x, list of weights
    return x,w 

# Define function to calculate period using Gaussian quadrature
def period(N, initial_position):
    points, weights = gaussxw(N)

    # Scale sample points to the interval [0, x0] (from [-1, 1])
    scaled_points = 0.5*initial_position*(points+1)
    scaled_weights = 0.5*initial_position*weights

    # Perform summation for Gaussian Quadrature
    integral = np.sum(scaled_weights/g(scaled_points, initial_position))

    # Calculate and return the period
    T = 4*integral

    return T

# Calculate the period for N=8 and N=16
T_N8 = period(8, x0)
T_N16 = period(16, x0)

# Estimate the fractional error
fractional_error = (np.abs(T_N16-T_N8)/T_N16)*100

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(f"The period for N = 8 is approximately {T_N8:.4f} seconds.")
print(f"The period for N = 16 is approximately {T_N16:.4f} seconds.")
print(f"The fractional error estimate between N = 8 and N = 16 is {fractional_error:.4f}%.")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~PART B~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''

# Define function to calculate the integrand (4/g_k) and weighted values (4w_k/g_k)
def integrand_values(N, initial_position):
    points, weights = gaussxw(N)
    scaled_points = 0.5*initial_position*(points+1)
    scaled_weights = 0.5*initial_position*weights

    g_values = g(scaled_points, x0)
    integrand = 4/g_values
    weighted_values = 4*scaled_weights/g_values

    return scaled_points, integrand, weighted_values

# Compute values for N = 8 and N = 16
points_8, integrand_8, weighted_8 = integrand_values(8, x0)
points_16, integrand_16, weighted_16 = integrand_values(16, x0)

# Plot the results
plt.figure(figsize=(12, 6))

# Plot for N=8
plt.subplot(1, 2, 1)
plt.plot(points_8, integrand_8, label=r'Integrand ($4/g_k$)', marker='o', color = plt.cm.tab20b(0), linewidth = 2)
plt.plot(points_8, weighted_8, label=r'Weighted ($4w_k/g_k$)', marker='x', color = plt.cm.tab20b(13), linewidth = 2)
plt.title(r'$N=8$', fontsize=20)
plt.xlabel(r'$x$', fontsize=17)
plt.ylabel('Value', fontsize=17)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=13)
plt.grid(True, color = 'lightgrey')

# Plot for N=16
plt.subplot(1, 2, 2)
plt.plot(points_16, integrand_16, label=r'Integrand ($4/g_k$)', marker='o', color = plt.cm.tab20b(5), linewidth = 2)
plt.plot(points_16, weighted_16, label=r'Weighted ($4w_k/g_k$)', marker='x', color = plt.cm.tab20b(21), linewidth = 2)
plt.title(r'$N=16$', fontsize=20)
plt.xlabel(r'$x$', fontsize=17)
plt.ylabel('Value', fontsize=17)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=13)
plt.grid(True, color = 'lightgrey')
plt.tight_layout()
plt.show()

'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~PART C~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''

# Calculate initial displacement for particle initially at rest which leads to speed c at x=0
x_c = c*np.sqrt(m/k)

# Create a range of x0 values and calculate corresponding T values
x0_range = np.linspace(1, 10*x_c, 20)
T_values = [period(16, x0_value) for x0_value in x0_range]

# Determine classical period
T_classical = 2*np.pi*np.sqrt(m/k)

# Plot results
plt.figure(figsize=(8, 6))
plt.plot(x0_range, T_values, label=r'Relativistic $T(x_0)$', marker='o', color = plt.cm.tab20b(0), linewidth = 2)
plt.axhline(T_classical, linestyle='--', label=r'Classical Limit $T=2\pi\sqrt{m/k}$', color = plt.cm.tab20b(5), linewidth = 2)
plt.plot(x0_range, 4 * x0_range / c, label=r'Relativistic Limit $T=4x_0/c$', linestyle='--', color = plt.cm.tab20b(13), linewidth = 2)

# Add labels, legend, gridlines, title
plt.xlabel(r'$x_0$ (m)', fontsize=17)
plt.ylabel(r'Period $T$ (s)', fontsize=17)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title(r'Period $T$ as a function of $x_0$', fontsize=20)
plt.legend(fontsize=14)
plt.grid(True, color = 'lightgrey')
plt.show()