from pyomo.environ import *
from pyomo.opt import SolverStatus, TerminationCondition
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv("Data.csv", header=None)
X = data.iloc[:, :-1].values # Features
y = data.iloc[:, -1].values   # Labels 

# Convert labels to binary: Iris-setosa = 1, others = -1
y = np.where(y == 'Iris-setosa', 1, -1)

# Parameters
d = X.shape[1]  # Number of features
M = 10  # Bound for w and b
epsilon = 1e-5  # Convergence threshold
max_iter  = 1000  # Maximum number of iterations

# Initialize w and b randomly
weights = np.random.uniform(-M, M, d)
bias = np.random.uniform(-M, M)

# Objective function: 1/2 * ||w||^2
def objective(z):
    w=z[:-1]
    return 0.5 * np.dot(w, w)

# Gradient of the objective function
def gradient(z):
    w=z[:-1]
    return np.append(w, 0)  # Gradient w.r.t. w is w, w.r.t. b is 0

# Solve linear programming problem to find v_k
def find_vk(z):
    model = ConcreteModel()
    
    # Decision variables: v = [w_v; b_v]
    model.weights_v = Var(range(d), bounds=(-M, M))
    model.bias_v = Var(bounds=(-M, M))
    
    # Objective: minimize gradient(z)^T v
    def objective_function(m):
        return sum(gradient(z)[j] * m.weights_v[j] for j in range(d)) + gradient(z)[-1] * m.bias_v
    
    model.objective = Objective(rule=objective_function, sense=minimize)
    
    # Constraints: y_i (w_v^T x_i + b_v) >= 1 for all i
    def constraint_function(m, i):
        return y[i] * (sum(m.weights_v[j] * X[i, j] for j in range(d)) + m.bias_v) >= 1
    
    model.constraints = Constraint(range(len(y)), rule=constraint_function)
    
    # Solve the problem
    solver = SolverFactory('glpk')
    results = solver.solve(model)
    
    # Extract the solution
    weights_v = np.array([model.weights_v[j]() for j in range(d)])
    bias_v = model.bias_v()
    
    return np.append(weights_v, bias_v)

# compute_tau_k using IPOPT
def compute_tau(z, d):
    model = ConcreteModel()
    
    # Decision variable: tau_k
    model.tau = Var(bounds=(0, 1))
    
    # Objective: minimize the objective function at z + tau * d
    def objective_function(m):
        return objective(z + m.tau * d)
    
    model.objective = Objective(rule=objective_function, sense=minimize)
    
    # Solve the problem
    solver = SolverFactory('ipopt')  # Use IPOPT solver
    results = solver.solve(model)
    
    # Extract the solution
    return model.tau()

# Optimization loop
z = np.append(weights, bias)
z_history = pd.DataFrame(columns=[f'weight_{j+1}' for j in range(d)] + ['bias'])
z_history.loc[0] = z  # Add initial solution to the DataFrame

for iteration in range(max_iter):
    # Find v_k
    v_k = find_vk(z)
    
    # Find direction d_k = v_k - z
    d_k = v_k - z
    
    # Compute_tau_k
    tau_k = compute_tau(z, d_k)
    
    # Update z
    z_new = z + tau_k * d_k
    z_history.loc[iteration + 1] = z_new
    
    # Check for convergence
    if np.linalg.norm(z_new - z) < epsilon:
        break
    
    z = z_new

# Extract optimal weights and bias
optimal_weights = z[:-1]
optimal_bias = z[-1]

# Predict function
def predict(X):
    return np.sign(np.dot(X, optimal_weights) + optimal_bias)

# Evaluate accuracy
y_pred = predict(X)
accuracy = np.mean(y_pred == y)
print(f"Final Accuracy: {accuracy:.4f}")
print(f"Final SVM Weights: {z}")
print(f"Final Bias: {optimal_bias}")

feature_importance = np.abs(optimal_weights)
top_2_indices = np.argsort(feature_importance)[-2:]  # Get two largest weights
print(f"Selected Features: {top_2_indices}")


def plot_decision_boundary(X, y, w, b, feature_indices):
    plt.figure(figsize=(8, 6))
    
    # Use only the two selected features
    X_vis = X[:, feature_indices]
    w_vis = w[feature_indices]  # Use the weights corresponding to selected features
    
    for label, color in zip([1, -1], ['#3d5a80', '#ee6c4d']):
        subset = X_vis[y == label]
        plt.scatter(subset[:, 0], subset[:, 1], label="Iris-setosa" if label == 1 else "Others", 
                    color=color, alpha=0.7, edgecolors='k')
    
    # Compute the decision boundary
    x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
    x_vals = np.linspace(x_min, x_max, 100)
    y_vals = -(w_vis[0] * x_vals + b) / w_vis[1]
    
    plt.plot(x_vals, y_vals, 'k-', linewidth=2, label='Decision Boundary')
    plt.xlabel(f'Feature {feature_indices[0] + 1}')
    plt.ylabel(f'Feature {feature_indices[1] + 1}')
    plt.title("Optimized SVM Decision Boundary")
    plt.legend()
    plt.show()

# Select the best two features and plot
plot_decision_boundary(X, y, optimal_weights, optimal_bias, top_2_indices)