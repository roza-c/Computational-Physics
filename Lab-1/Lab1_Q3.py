'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~PART B~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
import numpy as np

# Define function to integrate
def integrand(x):
    return 4/(1+x**2)

# Define Trapezoidal rule for integration 
def trapezoidal_rule(f, a, b, N):
    h = (b-a)/N                 # Slice width
    s1 = 0.5*f(a) + 0.5*f(b)    # Sum for the integral
    for k in range (1,N):       # Evaluate sum term-by-term
        s1 += f(a+k*h)
    result = h*s1               # Multiply sum by slice width
    return result  

# Define parameters
N = 4                           # Number of slices
a = 0.0                         # Lower limit of integration
b = 1.0                         # Upper limit of integration
pi = np.pi                      # Actual value of integral

# Evaluate integral using the Trapezoidal rule
result_trapezoidal = trapezoidal_rule(integrand, a, b, N)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("Integrating 4/(1+x**2) Using Trapezoidal and Simpson's Rule")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("The exact value of the integral is:", pi,"\n")
print("The result of the Trapezoidal rule is:", result_trapezoidal)
print("The difference between the exact value of the integral and the value obtained using the Trapezoidal rule is", pi-result_trapezoidal,"\n")

# Define Simpson's rule for integration 
def simpsons_rule(f, a, b, N):
    if N % 2 == 1:                  # Number of slices must be even
        print("Error: N must be an even number.")
        return None
    h = (b-a)/N                     # Slice width
    s2 = f(a) + f(b)                # Sum for the integral
    s2_term_1 = 0
    s2_term_2 = 0
    for k in range (1,((N//2)+1)):  # Evaluate sum term-by-term
        s2_term_1 += f(a+((2*k-1)*h))
    for k in range (1,(N//2)):      # Evaluate sum term-by-term
        s2_term_2 += f(a+(2*k*h))
    s2 = s2 + 4*s2_term_1 + 2*s2_term_2
    return((1/3)*h*s2)              # Multiply sum by slice width and 1/3

# Evaluate integral using Simpson's rule
result_simpson = simpsons_rule(integrand, a, b, N)

print("The result of the Trapezoidal rule is:", result_simpson)
print("The difference between the exact value and the value obtained using Simpson's rule is", pi-result_simpson)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")


'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~PART C~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
from time import time

# Parameters for Trapezoidal Rule
N = 4083                            # Number of slices
integration_reps = 101              # Number of times to repeat integral calculation
duration_trap = 0.0                 # Initialize duration of integral calculation to zero
total_duration_trap = 0             # Initialize variable to sum total durations of calculating integral

for i in range(integration_reps):
    start = time()                  # Initialize timer for manual method
    result_trapezoidal_timed = trapezoidal_rule(integrand, a, b, N)
    end = time()                    # Stopping timer
    duration_trap = end - start     # Calculate duration of computing integral
    total_duration_trap += duration_trap

# Calculate average time it took to compute the integral
average_duration_trap = total_duration_trap/(integration_reps-1)

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(" Time and # of slices to approximate integral with an error of O(10^-9)")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("For Trapezoidal method:")
print("N =",N, "slices were needed to approximate the integral with an error of",pi-result_trapezoidal_timed)
print("Computing this integral took", average_duration_trap, "seconds on average over", (integration_reps-1), "runs\n")

# Parameters for Simpson's Rule
M = 14                              # Number of slices
integration_reps = 101              # Number of times to repeat integral calculation
duration_simp = 0.0                 # Initialize duration of integral calculation to zero
total_duration_simp = 0             # Initialize variable to sum total durations of calculating integral

for i in range(integration_reps):
    start = time()                  # Initialize timer for manual method
    result_simpson_timed = simpsons_rule(integrand, a, b, M)
    end = time()                    # Stopping timer
    duration_simpson = end - start     # Calculate duration of computing integral
    total_duration_simp += duration_simpson

# Calculate average time it took to compute the integral
average_duration_simp = total_duration_simp/(integration_reps-1)
print("For Simpson's method:")
print("N =",M, "slices were needed to approximate the integral with an error of",pi-result_simpson_timed)
print("Computing this integral took", average_duration_simp, "seconds on average over", (integration_reps-1), "runs")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")


'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~PART D~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("Practical estimation of errors for Trapezoidal method")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

# Compute integrals using Trapezoidal method with N1 = 16 and N2 = 32 slices
N1 = 16
N2 = 32
I1 = trapezoidal_rule(integrand, a, b, N1)
I2 = trapezoidal_rule(integrand, a, b, N2)

# Estimate error using the practical estimation of errors method
practical_error_estimate = (1/3) * abs(I2 - I1)

print("The value of the integral using N1=", N1, "steps and the Trapezoidal method is:", I1)
print("The value of the integral using N2=", N2, "steps and the Trapezoidal method is:", I2)
print("The error using the practical estimation method is:", practical_error_estimate)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

