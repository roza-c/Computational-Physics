'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PART A ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
import numpy as np

def integrand(x):
    ''' Defines function to be integrated: x^(-1/2) / (1+e^(x)) '''
    return x**(-0.5) / (1+np.exp(x))

def mean_value_integration(N):
    ''' Performs mean value method for integration with N sample points '''
    # Generate random samples in [0, 1]
    x_samples = np.random.random(N)

    # Evaluate function at sampled points
    f_values = integrand(x_samples)

    # Compute and return integral estimate
    integral_estimate = np.mean(f_values)

    return integral_estimate

# Set parameters for simulation
num_samples = 10000 
num_repitions = 1000

# Perform integration
results = []
for i in range (num_repitions):
    result = mean_value_integration(num_samples)
    results.append(result)

# Compute and print mean of the 1000 results
mean_result = np.mean(results)
print(f'Mean of 1000 mean value integral estimates: {mean_result:.6e}')

'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PART B ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
def weighting_function(x):
    ''' Defines weighting function: w(x) = x^(-1/2) '''
    return x**(-0.5)

def integrand_weighted(x):
    ''' Defines original integrand divided by w(x): x^(-1/2) 1 / (1+e^(x)) '''
    return 1 / (1 + np.exp(x)) 

# Transformation x(z) to sample from the distribution p(x)
def transform_x(z):
    ''' Transforms uniform random numbers z in [0, 1) to sample from the distribution p(x) '''
    return z**2

def importance_sampling_integration(N=10000):
    ''' Performs importance sampling method for integration with N sample points '''
    # Generate uniform random samples z in [0, 1)
    z_samples = np.random.random(N)
    
    # Transform z to x using derived x(z) 
    x_samples = transform_x(z_samples)

    # Evaluate adjusted integrand at sampled points
    f_values = integrand_weighted(x_samples)

    # Compute and return integral estimate
    integral_estimate = np.mean(f_values)
    return integral_estimate

# Set parameters for simulation
num_repetitions = 1000

# Perform importance sampling
importance_results = []
for n in range(num_repetitions):
    result = importance_sampling_integration(num_samples)
    importance_results.append(result)

# Compute and print mean of the 1000 results
importance_mean_result = np.mean(importance_results)
print(f'Mean of 1000 importance sampling estimates: {importance_mean_result:.6e}')

'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PART C ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
import matplotlib.pyplot as plt

variance_mv = np.var(results)
variance_importance = np.var(importance_results)

# Histogram for the mean value method
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(results, bins=100, color='royalblue')
plt.title('Mean Value Method', fontsize=15)
plt.xlabel('Integral Estimate', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlim(0.79, 0.91)
plt.grid()
plt.text(0.98, 0.93, f'Variance: {variance_mv:.2e}', transform=plt.gca().transAxes,
         fontsize=17, verticalalignment='top', horizontalalignment='right')

# Histogram for the importance sampling method
plt.subplot(1, 2, 2)
plt.hist(importance_results, bins=100, color='forestgreen')
plt.title('Importance Sampling', fontsize=15)
plt.xlabel('Integral Estimate', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlim(0.41, 0.53)
plt.grid()
plt.text(0.98, 0.93, f'Variance: {variance_importance:.2e}', transform=plt.gca().transAxes,
         fontsize=17, verticalalignment='top', horizontalalignment='right')
plt.tight_layout()
plt.show()

'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PART D ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
def sharp_peak_integrand(x):
    ''' Defines new integrand: exp(-2*|x - 5|) '''
    return np.exp(-2 * np.abs(x-5))

def sharp_peak_weighting_function(x):
    ''' Defines weighting function w(x): (1 / sqrt(2*pi)) * exp(-(x-5)^2 / 2) '''
    return (1/np.sqrt(2*np.pi)) * np.exp(-0.5*(x-5)**2)

# Importance sampling method for sharp peak integral
def sharp_peak_importance_sampling(N=10000):
    ''' Performs importance sampling method for sharp peak integral with N sample points '''
    
    # Sample from weighting function w(x) using numpy's normal distribution (Mean=5, StdDev=1)
    x_samples = np.random.normal(loc=5, scale=1, size=N)

    # Keep only samples within interval [0, 10]
    x_samples = x_samples[(x_samples >= 0) & (x_samples <= 10)]

    # Evaluate integrand and weighting function at sampled points
    f_values = sharp_peak_integrand(x_samples)
    w_values = sharp_peak_weighting_function(x_samples)

    # Compute and return integral estimate
    integral_estimate = np.mean(f_values / w_values)
    return integral_estimate

# Define parameters for simulation
num_samples_sharp = 10000  
num_repetitions_sharp = 1000  

# Perform importance sampling 1000 times
sharp_peak_results = []
for _ in range(num_repetitions_sharp):
    result = sharp_peak_importance_sampling(num_samples_sharp)
    sharp_peak_results.append(result)

# Compute and print mean of the 1000 results
sharp_peak_mean_result = np.mean(sharp_peak_results)
print(f'Mean of 1000 importance sampling estimates for sharp peak: {sharp_peak_mean_result:.6e}')

variance_sp = np.var(sharp_peak_results)

# Plot histogram for sharp peak integral results
plt.figure(figsize=(8, 6))
plt.hist(sharp_peak_results, bins=100, color='mediumvioletred')
plt.title('Importance Sampling for Sharp Peak Integral', fontsize=15)
plt.xlabel('Integral Estimate', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid()
plt.text(0.98, 0.93, f'Variance: {variance_sp:.2e}', transform=plt.gca().transAxes,
         fontsize=17, verticalalignment='top', horizontalalignment='right')
plt.tight_layout()
plt.show()
