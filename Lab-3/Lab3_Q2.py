'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PART A ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
import numpy as np
from numpy.fft import rfft, irfft  
import matplotlib.pyplot as plt

# Load CSV file
data = np.genfromtxt("sp500.csv", delimiter=',', skip_header=1)

# Extract the business day number starting at 0 (index)
business_days = np.arange(len(data))

# Extract opening values (second column)
opening_values = data[:, 1]

# Plot the opening values against the business day number
plt.figure(figsize=(10, 6))
plt.plot(business_days, opening_values, color='darkblue', linewidth=1)
plt.title("S&P 500 Opening Values Over Business Days (2014-2019)")
plt.xlabel("Business Day Number")
plt.ylabel("Opening Value")
plt.grid(True)
plt.show()

'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PART B ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
# Store original length of real data 
original_length = len(opening_values)  

# Calculate FFT coefficients using rfft
fft_coefficients = rfft(opening_values)

# Invert FFT coefficients using irfft to recover original data
recovered_data = irfft(fft_coefficients, original_length)

# Check if reconstructed data matches the original data
test_result = np.allclose(opening_values, recovered_data, atol=10e-16)

# Display the test result
print(f"Test result: {'Success' if test_result else 'Failure'}")

# Plot original and reconstructed data to visually confirm
plt.figure(figsize=(10, 6))
plt.plot(range(len(opening_values)), opening_values, label='Original Data', color='blue')
plt.plot(range(len(recovered_data)), recovered_data, label='Reconstructed Data', linestyle='--', alpha=0.7, color='red')
plt.xlabel('Business Day Number')
plt.ylabel('Opening Value of S&P 500')
plt.title('Original vs Reconstructed Data')
plt.legend()
plt.grid(True)
plt.show()

'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PART C ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
# Calculate threshold frequency for 6-month period (approx. 126 business days)
cutoff_frequency = original_length // 126

# Set high-frequency coefficients to zero for low-pass filtering
filtered_fft = fft_coefficients.copy()
filtered_fft[cutoff_frequency:] = 0

# Perform inverse FFT with filtered coefficients
filtered_data = np.fft.irfft(filtered_fft, original_length)

# Plot original and filtered data
plt.figure(figsize=(12, 6))
plt.plot(range(original_length), opening_values, label='Original Data', color='darkblue')
plt.plot(range(original_length), filtered_data, label='Filtered Data (Long-Term Trend)', linestyle='--', color='red')
plt.xlabel('Business Day Number')
plt.ylabel('Opening Value of S&P 500')
plt.title('S&P 500 Opening Values: Original vs Long-Term Trend')
plt.legend()
plt.grid(True)
plt.show()
