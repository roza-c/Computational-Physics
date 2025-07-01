'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~PART B~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
# Import numpy libraries
import numpy as np
    
# Function to compute the relative error
def relative_error(x,y):
    return (x-y)/y

# Function to compute standard deviation using equation 1
def std_1(data):
    mean = np.mean(data)                            # Compute mean
    squared_diff = (data-mean)**2                   # Compute squared differences from mean
    variance = np.sum(squared_diff)/(len(data)-1)   # Compute variance
    return np.sqrt(variance)                        # Compute and return standard deviation

# Function to compute standard deviation using equation 2
def std_2(data):
    mean = np.mean(data)                            # Compute mean
    sum_of_squares = np.sum(data**2)                # Compute sum of squares of data
    variance = (sum_of_squares - len(data) * (mean**2))/(len(data)-1) # Compute variance       
    if variance<0:                                  # Check for negative square root   
        print("Warning: Negative variance encountered. Ceasing execution.")
        return None
    return np.sqrt(variance)                        # Compute and return standard deviation

# Load the dataset using numpy
data = np.loadtxt('cdata.txt')
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("Relative error of different standard deviation formulas of cdata.txt")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    
# Compute the correct standard deviation using numpy
correct_std = np.std(data, ddof=1)
print("The correct standard deviation is:", correct_std, "\n")
    
# Compute the standard deviations
std_dev_1 = std_1(data)
print("The standard deviation calculated using Formula 1 is:", std_dev_1)

std_dev_2 = std_2(data)
print("The standard deviation calculated using Formula 2 is:", std_dev_2,"\n")

# Compute the relative error for formula 1
relative_error_1 = relative_error(std_dev_1, correct_std)
print("Relative error for Formula 1: ", relative_error_1)

# Compute the relative error for formula 2
if std_dev_2 is not None:
    relative_error_2 = relative_error(std_dev_2, correct_std)
    print("Relative error for Formula 2: ", relative_error_2,"\n")

# Check which relative error is larger in magnitude.
larger_magnitude = max(abs(relative_error_1), abs(relative_error_2))
if larger_magnitude == abs(relative_error_1):
    relative_error_largest_magnitude = ('relative error 1')
if larger_magnitude == abs(relative_error_2):
    relative_error_largest_magnitude = ('relative error 2')

print("The relative error that is larger in magnitude is", relative_error_largest_magnitude)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~PART C~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
# Create two normally distributed sequences
data_1 = np.random.normal(loc=0., scale=1., size=2000)        # Mean = 0,   Sigma = 1, n = 2000
data_2 = np.random.normal(loc=1.e7, scale=1., size=2000)      # Mean = 1e7, Sigma = 1, n = 2000

# Compute correct standard deviations
correct_std_1 = np.std(data_1, ddof=1)
correct_std_2 = np.std(data_2, ddof=1)

# Compute the standard deviation using Formula 1
std_1_data1 = std_1(data_1)
std_1_data2 = std_1(data_2)

# Compute the standard deviation using Formula 2
std_2_data1 = std_2(data_1)
std_2_data2 = std_2(data_2)

# Compute the relative errors for Formula 1
relative_error_1_data1 = relative_error(std_1_data1, correct_std_1)
relative_error_1_data2 = relative_error(std_1_data2, correct_std_2)


print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("Relative error of different standard deviation formulas of random normal sequnces")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

print("For random normal sequnce 1 (mean = 0, sigma = 1, n = 2000):")
print("The correct standard deviation is:", correct_std_1)
print("The standard deviation calculated using Formula 1 is:", std_1_data1)
print("The standard deviation calculated using Formula 2 is:", std_2_data1)
print("Relative error for Formula 1 :", relative_error_1_data1)
# Compute the relative error for Formula 2
if std_2_data1 is not None: 
    relative_error_2_data1 = relative_error(std_2_data1, correct_std_1)
    print("Relative error for Formula 2:", relative_error_2_data1,"\n")

print("For random normal sequnce 2 (mean = 1e7, sigma = 1, n = 2000):")
print("The correct standard deviation is:", correct_std_2)
print("The standard deviation calculated using Formula 1 is:", std_1_data2)
print("The standard deviation calculated using Formula 2 is:", std_2_data2)
print("Relative error for Formula 1:", relative_error_1_data2)
# Compute the relative errors for Formula 2
if std_2_data2 is not None:
    relative_error_2_data2 = relative_error(std_2_data2, correct_std_2)
    print("Relative error for Formula 2:", relative_error_2_data2)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")


'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~PART D~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
# Function to shift the data
def std_shifted(data):
    # Calculate the mean and shift the data
    shift = np.mean(data)
    shifted_data = data - shift

    # Use Formula 2 on the shiftedered data 
    n = len(shifted_data)
    sum_sq_shifted = np.sum(shifted_data ** 2)
    variance_shifted = (sum_sq_shifted - n * 0**2) / (n - 1)

    # Return the standard deviation
    return np.sqrt(variance_shifted)

# Calculate standard deviations for both sequences using the shifted data
std_shifted_data1 = std_shifted(data_1)
std_shifted_data2 = std_shifted(data_2)

# Calculate the "correct" standard deviations using numpy for comparison
std_correct_data1 = np.std(data_1, ddof=1)
std_correct_data2 = np.std(data_2, ddof=1)

# Calculate relative errors for the centered method
relative_error_data1 = (std_shifted_data1 - std_correct_data1) / std_correct_data1
relative_error_data2 = (std_shifted_data2 - std_correct_data2) / std_correct_data2

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("Relative error of shifted standard deviation formula")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

print("For random normal sequnce 1 (mean = 0, sigma = 1, n = 2000):")
print("The correct standard deviation is:", std_correct_data1)
print("The standard deviation calculated using Formula 2 is:", std_shifted_data1)
# Compute the relative error for Formula 2
if std_shifted_data1 is not None: 
    relative_error_data1 = (std_shifted_data1 - std_correct_data1) / std_correct_data1
    print("Relative error for Formula 2:", relative_error_data1,"\n")

print("For random normal sequnce 2 (mean = 1e7, sigma = 1, n = 2000):")
print("The correct standard deviation is:", std_correct_data2)
print("The standard deviation calculated using Formula 2 is:", std_shifted_data2)
# Compute the relative error for Formula 2
if std_shifted_data2 is not None: 
    relative_error_data2 = (std_shifted_data2 - std_correct_data2) / std_correct_data2
    print("Relative error for Formula 2:", relative_error_data2)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

