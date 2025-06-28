PHY407 Formal Lab 1: Numerical Errors and Simple Numerical Integration

This repository contains the Python code and results for the formal lab report on numerical errors and integration, as part of the PHY407 course. The lab explores fundamental concepts in computational physics, including the stability of numerical algorithms, the nature of roundoff error, and the efficiency and accuracy of different numerical integration techniques.
Description

This lab is divided into three main investigations:

    Numerical Issues in Standard Deviation Calculations: This section compares two mathematically equivalent formulas for calculating standard deviation. It investigates how the choice of algorithm (a "one-pass" vs. a "two-pass" method) can lead to significant differences in numerical precision, especially when dealing with datasets that have a large mean and small variance.

    Exploring Roundoff Error: This part examines the effects of floating-point arithmetic errors by comparing two forms of the polynomial p(u) = (1 - u)^8. By plotting the expanded form versus the factored form near u=1, the exercise demonstrates how catastrophic cancellation can introduce significant numerical noise.

    Trapezoidal and Simpson's Rules for Integration: This section involves implementing and comparing the Trapezoidal and Simpson's rules to evaluate the definite integral of 4 / (1 + x^2) from x=0 to 1, which equals pi. The analysis focuses on the rate of convergence and computational efficiency of each method to achieve a desired level of accuracy.

Technologies Used

    Python 3

    NumPy: For numerical operations, array handling, and random number generation.

    Matplotlib: For generating plots and histograms.

Files in this Repository

    Lab1_Q1.py: Python script for Question 1, analyzing standard deviation formulas.

    Lab1_Q2.py: Python script for Question 2, investigating roundoff error in polynomials.

    Lab1_Q3.py: Python script for Question 3, implementing and comparing integration rules.

    cdata.txt: The input data file containing Michelsen's speed of light measurements, required to run Lab1_Q1.py.

    PHY407_FormalLab1_Report.pdf (Optional): The final written lab report including analysis, figures, and answers to the questions.

    README.md: This file.

How to Run the Code

To run the code and reproduce the results, you will need a Python environment with NumPy and Matplotlib installed.

    Clone or download this repository to your local machine.

    Ensure the cdata.txt file is in the same directory as the Python scripts.

    Navigate to the project directory in your terminal.

    Run each script from the command line:

# To run the analysis for Question 1
python Lab1_Q1.py

# To run the analysis for Question 2
python Lab1_Q2.py

# To run the analysis for Question 3
python Lab1_Q3.py

The scripts will print the required numerical results to the console and generate plots as specified in the lab questions.