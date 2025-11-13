# Support Vector Machine Optimisation from Scratch

[View Full Report](svm-optimisation-report.pdf)  
[View Python Code](svm-optimisation-code.py)

**Tools:** Python, Pyomo, NumPy, GLPK, IPOPT  
**Techniques:** Constrained optimisation, feasible direction method, hard-margin SVM, soft-margin SVM, numerical optimisation

## Overview
This project develops a Support Vector Machine (SVM) classifier entirely from first principles using mathematical optimisation techniques.  
The objective was to translate the theoretical hard-margin SVM formulation into a working optimisation algorithm without relying on machine-learning libraries.

## Approach
- Implemented the **hard-margin SVM** objective  
  \[
  \min \frac{1}{2} \|w\|^2 \quad \text{s.t. } y_i(w^T x_i + b) \ge 1
  \]
  using Pyomo to model constraints and GLPK as the linear solver.
  
- Applied a **feasible direction algorithm**, iteratively updating  
  - feasible point \( v_k \)  
  - descent direction \( d_k = v_k - z_k \)  
  - optimal step size \( \tau_k \) using IPOPT

- Added **soft-margin extension** using slack variables and penalty parameter \( C \).

- Evaluated classification performance on the Iris dataset using a binary transformation:
  - Setosa = +1  
  - Non-Setosa = –1

- Visualised the optimised decision boundary using the two most important features.

## Results
- Achieved **100% classification accuracy** distinguishing Setosa vs non-Setosa.  
- Final optimised weight vector:  
  \[
  w = [-0.046, \; 0.522, \; -1.003, \; -0.464]
  \]
- Bias term:  
  \[
  b = 1.451
  \]
- Identified **petal length** and **sepal width** as the most influential features.
- Convergence achieved with tolerance \( \epsilon = 1e^{-5} \), using bounded optimisation with \( M = 10 \).

## Impact
This project demonstrates the ability to:
- Translate mathematical ML formulations into working optimisation algorithms  
- Build an SVM classifier **without scikit-learn**  
- Apply numerical optimisation techniques (GLPK, IPOPT, Pyomo)  
- Analyse convergence, constraints, and algorithmic performance  
- Communicate results with both technical depth and interpretability

The project showcases strong understanding of optimisation theory and its practical implementation for machine-learning models.
