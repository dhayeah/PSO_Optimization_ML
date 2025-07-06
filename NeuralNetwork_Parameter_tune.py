import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from pyswarm import pso

# Generate larger synthetic regression dataset with reduced noise
X, y = make_regression(n_samples=500, n_features=1, noise=3, random_state=42)

# Scale features and target, clip extreme values for stability
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
X_scaled = np.clip(X_scaled, -5, 5)  # Clip to avoid extreme values
y_scaled = np.clip(y_scaled, -5, 5)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Store PSO fitness history
fitness_history = []

# Objective function for PSO: Minimize MSE
def objective_function(params):
    learning_rate, hidden_layer_size = params
    hidden_layer_size = int(hidden_layer_size)
    try:
        model = MLPRegressor(hidden_layer_sizes=(hidden_layer_size,), learning_rate_init=learning_rate,
                            max_iter=1500, tol=1e-3, random_state=42, solver='sgd',
                            momentum=0.95, learning_rate='adaptive', n_iter_no_change=3)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
    except Exception as e:
        print(f"Training failed: {e}")
        mse = 1e10
    fitness_history.append(mse if np.isfinite(mse) else 1e10)
    return mse if np.isfinite(mse) else 1e10

# PSO optimization with tighter bounds
lb = [0.001, 3]   # Lower bounds: learning rate, hidden layer size
ub = [0.003, 10]  # Upper bounds: reduced learning rate for SGD stability
best_params, best_mse = pso(objective_function, lb, ub, swarmsize=8, maxiter=15)

# Train final model
best_lr, best_hls = best_params
best_hls = int(best_hls)
model = MLPRegressor(hidden_layer_sizes=(best_hls,), learning_rate_init=best_lr,
                    max_iter=1500, tol=1e-3, random_state=42, solver='sgd',
                    momentum=0.95, learning_rate='adaptive', n_iter_no_change=3)
model.fit(X_train, y_train)
y_pred_scaled = model.predict(X_test)

# Inverse transform for visualization
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
X_test_orig = scaler_X.inverse_transform(X_test)

# Visualization for PPT
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(fitness_history, 'o-', color='#1f77b4')
plt.title('PSO Convergence (SGD)')
plt.xlabel('Iteration')
plt.ylabel('MSE')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(X_test_orig, y_test_orig, color='black', label='Actual', alpha=0.6)
plt.scatter(X_test_orig, y_pred, color='#ff7f0e', label='Predicted', alpha=0.6)
plt.title(f'NN Predictions (LR={best_lr:.3f}, HLS={best_hls})')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('pso_nn_sgd_stable_v2.png', dpi=300)
plt.close()

# Print results
print(f"Best Parameters: Learning Rate = {best_lr:.3f}, Hidden Layer Size = {best_hls}")
print(f"Best MSE (scaled): {best_mse:.3f}")
