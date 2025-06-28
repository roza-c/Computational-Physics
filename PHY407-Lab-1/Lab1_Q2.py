'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~PART A~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
import numpy as np
import matplotlib.pyplot as plt

# Define the function p(u)
def p(u):
    return (1 - u)**8

# Define the function q(u)
def q(u):
    return 1 - 8*u + 28*u**2 - 56*u**3 + 70*u**4 - 56*u**5 + 28*u**6 - 8*u**7 + u**8

# Generate 500 points in the range 0.98 < u < 1.02
u_values = np.linspace(0.98, 1.02, 500)

# Calculate p(u) and q(u)
p_values = p(u_values)
q_values = q(u_values)


# Plot p(u) and q(u) on the same graph
plt.figure(figsize=(8,6))
plt.plot(u_values, p_values, label=r'$p(u)=(1-u)^8$', color='darkviolet', linewidth='1.75')
plt.plot(u_values, q_values, label=r'$q(u)=1-8u+28u^2-...+u^8$', color='darkgreen', linewidth='0.75')
plt.xlabel('u')
plt.ylabel('Value')
plt.title('Comparison of p(u) and q(u) near u=1')
plt.legend()
plt.grid(True)
plt.show()

'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~PART B~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("Standard deviation from histogram compared with estimate from equation (3)")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

# Calculate difference between p(u) and q(u)
difference_p_q = p_values - q_values

# Calculate standard deviation of the differences using numpy
std_numpy = np.std(difference_p_q)
print("The standard deviation of the distribution using numpy std is:", std_numpy, "\n")

# Plot the difference p(u) - q(u)
plt.figure(figsize=(14,6))

# Plot of p(u) - q(u)
plt.subplot(1, 2, 1)
plt.plot(u_values, difference_p_q, color='darkgreen', linewidth=1)
plt.xlabel('u')
plt.ylabel('p(u) - q(u)')
plt.title('Difference between p(u) and q(u) near u=1')
plt.grid(True)

# Histogram of p(u) - q(u)
plt.subplot(1, 2, 2)
plt.hist(difference_p_q, bins=30, color='darkviolet', alpha=0.5, edgecolor='black')
plt.xlabel('p(u) - q(u)')
plt.ylabel('Frequency')
plt.title('Histogram of p(u) - q(u) near u=1')

# Display both plots
plt.tight_layout()
plt.show()

# Define a function to compute coefficients of expansion q(u)
def calculate_x_values(u):
    x_1 = 1
    x_2 = -8 * u
    x_3 = 28 * u**2
    x_4 = -56 * u**3
    x_5 = 70 * u**4
    x_6 = -56 * u**5
    x_7 = 28 * u**6
    x_8 = -8 * u**7
    x_9 = 1 * u**8

    x_array = [x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9]
    return x_array

# Determine coefficients for expansion of q(1)
q_1_array = calculate_x_values(1)

print("The expansion coefficients of q(1) are:")
expansion_coefficients = [f"q_{i}: {q_1_array[i]}" for i in range(len(q_1_array))]
print(", ".join(expansion_coefficients), "\n")

# Define error constant C
C = 10**(-16)

# Compute mean of expension terms of q(1)
q_1_mean = np.mean(q_1_array)
# Square each expension term of q(1)
squared_q_1_array = [x**2 for x in q_1_array]
# Take the square root of the mean of the values squared
sqrt_q_mean_squared = np.sqrt(np.mean(squared_q_1_array))
# Determine number of terms in expansion
q_N = len(q_1_array)

# Compute statistical quantity sigma
sigma_estimate_3 = (C*np.sqrt(q_N)*sqrt_q_mean_squared)
print("The estimate obtained using equation 3 is:", sigma_estimate_3)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~PART C~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''

# Generate 500 points in the range 0.98 < u < 1.02
u_values_small_range = np.linspace(0.980, 0.984, 500)

# Calculate p(u) and q(u)
p_values_small_range = p(u_values_small_range)
q_values_small_range = q(u_values_small_range)

# Calculate difference between p(u) and q(u)
small_difference_p_q = p_values_small_range - q_values_small_range

# Calculate number of terms in the sum 
N = len(small_difference_p_q)

# Calculate mean of squared difference values
small_mean_squared_difference = np.mean(small_difference_p_q**2)

# Calculate fractional error: abs(p - q) / abs(p)
fractional_error = np.abs(p(u_values_small_range) - q(u_values_small_range)) / np.abs(p(u_values_small_range))

# Plot fractional error
plt.plot(u_values_small_range, fractional_error, linewidth = '0.75', color='darkgreen', alpha=0.9, label='Fractional Error')
plt.axhline(1.0, color='darkviolet', linestyle='--',label='100% Error')
plt.xlabel('u')
plt.ylabel('Fractional Error (|p - q| / |p|)')
plt.title('Fractional Error as u approaches 1.0')
plt.legend()
plt.grid(True)
plt.show()

print("Fractional error values that are greater than or equal to 1:\n")

# Print fractional error values greater than or equal to 1
for i, error in enumerate(fractional_error):
    if error >= 1:
        print("u =", u_values_small_range[i], "fractional error =", error)


print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("Fractional error on q(u) using equation (4)")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

# Loop over the u values and perform the calculation for each
u_values_loop = [0.95, 0.97, 0.98, 0.984, 0.992, 1.1, 2.0]

for u in u_values_loop:
    # Determine coefficients for expansion of q(u)
    q_array = calculate_x_values(u)
    
    print(f"The expansion coefficients of q({u}) are:")
    expansion_coefficients = [f"q_{i}: {q_array[i]}" for i in range(len(q_array))]
    
    # Print the first 5 coefficients on the first line
    print("\t".join(expansion_coefficients[:5]))  # Print first 5 terms
    
    # Print the remaining coefficients in chunks of 4 per line
    for i in range(5, len(expansion_coefficients), 4):
        print("\t".join(expansion_coefficients[i:i+4]))  # Print in groups of 4

    # Compute mean of expansion terms of q(u)
    q_mean = np.mean(q_array)
    
    # Square each expansion term of q(u)
    squared_q_array = [x**2 for x in q_array]
    
    # Take the square root of the mean of the values squared
    sqrt_q_mean_squared = np.sqrt(np.mean(squared_q_array))
    
    # Determine number of terms in expansion
    q_N = len(q_array)
    
    # Compute statistical quantity sigma
    sigma_estimate = (C / np.sqrt(q_N)) * (sqrt_q_mean_squared / q_mean)
    
    # Print the sigma estimate for the current value of u
    print(f"The fractional error obtained using equation 4 for u = {u} is {sigma_estimate}")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~PART D~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("Standard deviation of f = u**8/((u**4)*(u**4))")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

# Define a function to compute coefficients of expansion q(u)
def new_function(u):
    return u**8/((u**4)*(u**4))

# new function evaluated at near-1 values of u
new = new_function(u_values)

# standrd deviation computed with numpy
std_error = np.std(new-1)

# compute sigma = C*x
error_estimate = C*(new_function(1))

print("The standard deviation of f using numpy std is:", std_error)
print("The error estimate using sigma = Cx is:", error_estimate)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")



# Plot the error (f - 1) versus u
plt.figure(figsize=(8, 6))
plt.plot(u_values, (new-1), label='f(u) - 1', color='darkgreen', linewidth='0.75')
plt.axhline(0, color='darkviolet', linestyle='--', label='Expected Value (0)', linewidth='1.5')
plt.title("Plot of f(u) - 1 versus u")
plt.xlabel("u")
plt.ylabel("f(u) - 1 (Error)")
plt.legend()
plt.grid(True)
plt.show()


