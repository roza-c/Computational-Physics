Lab 1: Numerical Errors and Simple Integration

This project explores fundamental concepts in numerical computation, including the stability of different algorithms for statistical calculations, the impact of machine roundoff error, and a comparison of numerical integration methods. 

---
### Objectives

* **Question 1: Standard Deviation Errors**
    * Implement and compare a one-pass and a two-pass formula for calculating standard deviation to investigate their numerical stability, using both supplied data and generated random numbers.

* **Question 2: Roundoff Error**
    * Investigate the effects of roundoff error by comparing the numerical results of a polynomial in its factored form, p(u) = (1-u)^8, versus its expanded form near u=1.

* **Question 3: Integration Rules**
    * Evaluate the integral of 4 / (1 + x^2) from 0 to 1. Compare the accuracy and efficiency of the Trapezoidal Rule versus Simpson's Rule to achieve a target error of O(10^-9). 

---
### Files in this Repository

* Lab1_Q1.py
* Lab1_Q2.py
* Lab1_Q3.py
* cdata.txt
* README.md