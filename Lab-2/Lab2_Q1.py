'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~PART A~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
# Function to calculate the nth Hermite polynomial H_n(x)
import numpy as np

def hermite_polynomial(n, x):
    # Check if integer n ≥ 0
    if n < 0:
        print("Error: n must be a non-negative integer.")
        return None

    # Base case 1: first Hermite polynomial H_0(x) = 1
    elif n == 0:
        # Returns an array of ones with the same shape and type as array x      
        return np.ones_like(x)    
    
    # Base case 2: second Hermite polynomial H_1(x) = 2x 
    elif n == 1:    
        return 2*x
    
    # Recursion: nth Hermite polynomial H_n(x) 
    else:   # If n > 1
        
        H_n_minus_2 = np.ones_like(x)   # H_0
        H_n_minus_1 = 2*x               # H_1
        
        # function calls itself to calculate nth Hermite polynomial            
        for i in range(2, n+1):
            H_n = (2 * x * H_n_minus_1) - (2 * (i-1) * H_n_minus_2)
            H_n_minus_2 = H_n_minus_1
            H_n_minus_1 = H_n

        return H_n

'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~PART B~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
# Plot harmonic oscillator wavefunctions for n = 0, 1, 2, 3 in range -4 ≤ x ≤ 4

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from math import factorial, sqrt, pi

# Define quantum harmonic oscillator wavefunction
def psi_n(x, n):

    # Calculate nth Hermite polynomial
    H_n = hermite_polynomial(n, x)

    return (1 / (sqrt(2**n * factorial(n) * sqrt(pi)))) * np.exp(-x**2/2) * H_n

# Create x-values in the range -4 ≤ x ≤ 4
x_values = np.linspace(-4, 4, 400)

# Plot wavefunctions for n = 0, 1, 2, 3
plt.figure(figsize=(10,6))

colour = iter(cm.tab20b(np.linspace(0, 1, 4)))
for n in range(4):
    plot_colour = next(colour)
    plt.plot(x_values, psi_n(x_values, n), label=f'$ψ_{n}(x)$', c=plot_colour, linewidth=2)

# Add title, labels, gridlines, legend to plot
plt.title(r'Harmonic Oscillator Wavefunctions for $n = 0, 1, 2, 3$', fontsize=20)
plt.xlabel(r'$x$', fontsize=20)
plt.ylabel(r'$\psi_{n}(x)$', fontsize=20)
plt.legend(fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True)
plt.show()

'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~PART C~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
# Evaluate potential energy of QM harmonic oscillator using Gaussian quadrature

# Use gaussxw.py to find sample points and weights
from pylab import *
def gaussxw(N):

    # Initial approximation to roots of the Legendre polynomial
    a = linspace(3,4*N-1,N)/(4*N+2)
    x = cos(pi*a+1/(8*N*N*tan(a)))

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
        delta = max(abs(dx))

    # Calculate the weights
    w = 2*(N+1)*(N+1)/(N*N*(1-x*x)*dp*dp)

    # Return width of sample points x, list of weights
    return x,w 

# Use change of variables to deal with integral over infinite range
# x = tan(z), dx = dz/cos^2(z), with −π/2 ≤ z ≤ π/2​

# Function to define the integrand
def integrand(z, n):
    x = np.tan(z)                   # x = tan(z)
    psi = psi_n(x, n)               # QMHO wavefunction
    dz = (x**2) * (np.abs(psi)**2)  # dz
    dx = dz/(np.cos(z)**2)          # dx = dz/cos^2(z)
    return dx                       # Return converted integrand

# Set number of sample points for Gaussian quadrature
N = 100

# Write function using Gaussian quadrature for integration over (-pi/2, pi/2)
def gaussian_quadrature(n, N):
    # Compute 100 sample points and weights
    z_points, z_weights = gaussxw(N)

    # Rescale from (-1, 1) to (-pi/2, pi/2)
    z_points_rescaled = 0.5 * z_points * pi
    z_weights_rescaled = 0.5 * z_weights * pi

    # Perform summation for Gaussian Quadrature
    integral = np.sum(z_weights_rescaled * integrand(z_points_rescaled, n))

    # Return result of integral using transformed points and weights
    return integral

# Initialize empty array for potential energy values <x^2>
potential_energies = []

# Calculate <x^2> for each n from 0 to 10
for n in range(11):
    energy = gaussian_quadrature(n, N)
    potential_energies.append(energy)

# Print out the results, rounding to two decimal places
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

for n, energy in enumerate(potential_energies):
    print(f"For n = {n}, the potential energy is {energy:.2f}")

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")