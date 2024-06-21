import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Example synthetic data (replace with real data)
returns = pd.DataFrame({
    'Asset 1': [0.03, 0.01, 0.02, 0.04, 0.05, 0.01, 0.02, 0.03, 0.04, 0.05, 0.01, 0.02],
    'Asset 2': [0.02, 0.03, 0.04, 0.01, 0.02, 0.05, 0.03, 0.04, 0.01, 0.02, 0.03, 0.04],
    'Asset 3': [0.01, 0.02, 0.03, 0.04, 0.03, 0.02, 0.01, 0.02, 0.03, 0.04, 0.02, 0.03],
    'Asset 4': [0.04, 0.01, 0.02, 0.03, 0.01, 0.04, 0.05, 0.02, 0.03, 0.01, 0.04, 0.02],
    'Asset 5': [0.05, 0.03, 0.04, 0.01, 0.02, 0.03, 0.04, 0.05, 0.01, 0.02, 0.03, 0.04],
    'Asset 6': [0.02, 0.04, 0.03, 0.05, 0.01, 0.02, 0.03, 0.04, 0.05, 0.01, 0.02, 0.03]
})

# Calculate mean returns and covariance matrix
mean_returns = returns.mean()
cov_matrix = returns.cov()

# Market weights (example, should be based on market cap)
market_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.1, 0.1])

# Risk aversion coefficient (example)
delta = 2.5

# Implied equilibrium returns
pi = delta * np.dot(cov_matrix, market_weights)

# Investor views
# Absolute view: Asset 1 expected return is 8%
# Relative views: Asset 2 will outperform Asset 3 by 3%, and Asset 4 will outperform Asset 5 by 2%
P = np.array([
    [1, 0, 0, 0, 0, 0],
    [0, 1, -1, 0, 0, 0],
    [0, 0, 0, 1, -1, 0]
])
Q = np.array([0.08, 0.03, 0.02])

# Uncertainty in the views (diagonal matrix)
Omega = np.diag([0.0001, 0.0001, 0.0001])

# Combine the expected returns
tau = 0.025
M_inverse = np.linalg.inv(np.linalg.inv(tau * cov_matrix) + np.dot(np.dot(P.T, np.linalg.inv(Omega)), P))
E_R = np.dot(M_inverse, (np.dot(np.linalg.inv(tau * cov_matrix), pi) + np.dot(np.dot(P.T, np.linalg.inv(Omega)), Q)))

# Portfolio optimization (using scipy.optimize)
def portfolio_variance(weights, cov_matrix):
    return weights.T @ cov_matrix @ weights

def portfolio_return(weights, returns):
    return weights.T @ returns

def objective_function(weights, cov_matrix, returns, risk_aversion):
    return risk_aversion * portfolio_variance(weights, cov_matrix) - portfolio_return(weights, returns)

constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
bounds = tuple((0, 1) for _ in range(len(mean_returns)))
initial_guess = market_weights

result = minimize(objective_function, initial_guess, args=(cov_matrix, E_R, delta), method='SLSQP', bounds=bounds, constraints=constraints)

# Optimal weights
optimal_weights = result.x
print("Optimal Weights: ", optimal_weights)
