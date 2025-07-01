'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~PART A~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
import numpy as np

# Define function f(x) = e^(-x^2) whose derivative will be taken
def f(x):
    return np.exp(-x**2)

# Define third derivative of function f'''(x) = -4x(3-2x^2)e^(-x^2)
def f_third_derivative(x):
    return -4.*x*(3. - 2.*x**2)*np.exp(-x**2)

# Define function to calculate central difference approximation
def central_difference(f, x, h):
    return (f(x+(h/2.)) - f(x-(h/2.)))/h

# Define range of h values from 10^-16 to 10^0
h_values = [10.**(-16 + i) for i in range(17)]

# Create empty list to hold derivative results
results = []

# Compute numerical derivative for each h at x=1/2
x = 0.5
for h in h_values:
    derivative = central_difference(f, x, h)    # calculate derivative
    results.append((h, derivative))             # add as tuple to list

# Print the results
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(f"For h = {results[0][0]:.1e}, f'(x) = {results[0][1]}")
for h, derivative in results[1:]:
    print(f"For h = {h:.1e}, f'(x) ≈ {derivative}")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~PART B~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''

# Define function to compute first derivative f'(x) = (-2x)e^(-x^2)
def f_first_derivative(x):
    return -2*x*np.exp(-x**2)

# Calculate the analytical derivative at x = 1/2
analytical_derivative = f_first_derivative(x)

# Calculate absolute value of relative error for each numerical derivative
relative_errors = [(h, abs((numerical_result-analytical_derivative)/analytical_derivative)) for h, numerical_result in results]

# Find the h values that yields the minimum error
# Key specifies a function lambda that determines the value to compare for each element in the list
# Tells min() to find the tuple in relative_error where the error (second element) is the smallest
min_error_h_val, min_error = min(relative_errors, key=lambda x: x[1])

# Set machine precision constant
C = 10**(-16)

# Compute theoretical optimal minimum error
theoretical_error = ((9/8)*(C**2)*((f(x))**2)*abs(f_third_derivative(x)))**(1/3)

# Compute theoretical optimal h value yielding the minimum error
optimal_h_val = (24*C * np.abs(f(x) / f_third_derivative(x)))**(1/3)

# Print the results
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(f"The analytical derivative of f(x) = exp(-x^2) at x = 1/2 is approximately {analytical_derivative}.\n")
print(f"The h-value which yields the minimum absolute relative error of the numerical derivative is\nh = {min_error_h_val:.1e}, with a relative error of approximately {min_error}.\n")
print(f"The theoretical optimal error on the estimate of the derivative is ε = {theoretical_error}.\n")
print(f"The theoretical optimal h value which will produce the minimum error is h = {optimal_h_val:.1e}.")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~PART C~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''

# Define the second derivative of the function f''(x) = (4x^2-2)e^(-x^2)
def f_second_derivative(x):
    return ((4*x**2)-2)*np.exp(-x**2)

# Define function to calculate forward difference approximation
def forward_difference(f, x, h):
    return (f(x + h) - f(x))/h

# Compute numerical derivatives using forward difference for each h at x=1/2
forward_diff_results = [(h, forward_difference(f, x, h)) for h in h_values]

# Print the results
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
for h, derivative in forward_diff_results:
    print(f"For h = {h:.1e}, f'(x) ≈ {derivative}")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

# Calculate absolute value of relative error for each numerical derivative
fwd_relative_errors = [(h, abs((fwd_result-analytical_derivative)/analytical_derivative)) for h, fwd_result in forward_diff_results]

# Find the h values that yields the minimum error
fwd_min_error_h_val, fwd_min_error = min(fwd_relative_errors, key=lambda x: x[1])

# Compute theoretical total error for forward difference
fwd_theoretical_error = np.sqrt(4*C*abs(f(x)*f_second_derivative(x)))

# Compute optimal theoretical h value for forward difference
fwd_optimal_h_val = np.sqrt(4 * C * np.abs(f(x) / f_second_derivative(x)))

# Print the results
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(f"The h-value which yields the minimum absolute relative error of the numerical derivative is\nh = {fwd_min_error_h_val:.1e}, with a relative error of approximately {fwd_min_error}.\n")
print(f"The theoretical optimal error on the estimate of the derivative is ε = {fwd_theoretical_error}.\n")
print(f"The theoretical optimal h value which will produce the smallest error is h = {fwd_optimal_h_val:.1e}.")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~PART D~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
import matplotlib.pyplot as plt

# Ploting the results
plt.figure()

# Use logarithmic axes
plt.loglog(h_values, [error[1] for error in relative_errors], label="Central Difference", marker='x', color = plt.cm.tab20b(14), linewidth = 2.25)
plt.loglog(h_values, [error[1] for error in fwd_relative_errors], label="Forward Difference", marker='o', color = plt.cm.tab20b(4), linewidth = 2.25)

# Label axes, add title, legend, and grid
plt.xlabel('h', fontsize=17)
plt.ylabel('Absolute Relative Error', fontsize=17)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('Relative Error of Numerical Derivative (Central vs Forward Difference)', fontsize=20)
plt.legend(fontsize=14)
plt.grid(True, color = 'lightgrey')
plt.show()

'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~PART F~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''

# Define the function g(x)=exp(2x)
def g(x):
    return np.exp(2*x)

# Define the central difference method for first 5 derivatives of g(x)
def central_difference_g(f, x, h, order):
    if order == 1:
        return (f(x + h/2) - f(x - h/2)) / h
    elif order == 2:
        return (f(x + h) - 2*f(x) + f(x - h)) / h**2
    elif order == 3:
        return (f(x + 1.5*h) - 3*f(x + h/2) + 3*f(x - h/2) - f(x - 1.5*h)) / h**3
    elif order == 4:
        return (f(x + 2*h) - 4*f(x + h) + 6*f(x) - 4*f(x - h) + f(x - 2*h)) / h**4
    elif order == 5:
        return (f(x + 2.5*h) - 5*f(x + 1.5*h) + 10*f(x + h/2) - 10*f(x - h/2) + 5*f(x - 1.5*h) - f(x - 2.5*h)) / h**5
    else:
        return None

# Set point and step size for computing the derivative
x = 0     
h = 10**(-6)

# Calculate the first 5 derivatives using the central difference method
g_derivatives = [central_difference_g(g, x, h, order) for order in range (1, 6)]

# Compute the analytical derivatives for g(x) = exp(2x) at x = 0
# g'(x) = 2*exp(2x), g''(x) = 4*exp(2x), ..., g^(n)(x) = 2^n * exp(2x)
g_analytical_derivatives = [2**order * np.exp(2*x) for order in range (1, 6)]

# Print the results
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("Central Difference Derivative\t Analytical Derivative")
print(f"g'(0) ≈ {g_derivatives[0]}\t g'(0) = {g_analytical_derivatives[0]}")
print(f"g''(0) ≈ {g_derivatives[1]}\t g''(0) = {g_analytical_derivatives[1]}")
print(f"g'''(0) ≈ {g_derivatives[2]}\t g'''(0) = {g_analytical_derivatives[2]}")
print(f"g''''(0) ≈ {g_derivatives[3]}\t g''''(0) = {g_analytical_derivatives[3]}")
print(f"g'''''(0) ≈ {g_derivatives[4]}\t g'''''(0) = {g_analytical_derivatives[4]}")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")