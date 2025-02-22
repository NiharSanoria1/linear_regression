from lin_reg import LinearRegression

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate synthetic dataset
np.random.seed(42)

# Generate features
n_samples = 1000
X = np.random.randn(n_samples, 3)  # 3 features
# Create true relationships
true_weights = [2, -1, 3]
true_bias = 4

# Generate target with some noise
y = np.dot(X, true_weights) + true_bias + np.random.normal(0, 1, n_samples)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LinearRegression(learning_rate=0.01, no_of_iterations=1000)
model.fit(X_train_scaled, y_train)

# Make predictions
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Calculate metrics
train_score = model.score(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)
train_mse = np.mean((y_train - y_train_pred) ** 2)
test_mse = np.mean((y_test - y_test_pred) ** 2)

# Print results
print(f"Training R² Score: {train_score:.4f}")
print(f"Testing R² Score: {test_score:.4f}")
print(f"Training MSE: {train_mse:.4f}")
print(f"Testing MSE: {test_mse:.4f}")
print("\nLearned weights:", model.weights)
print("True weights:", true_weights)
print("\nLearned bias:", model.bias)
print("True bias:", true_bias)

# Plotting
plt.figure(figsize=(15, 5))

# Plot 1: Loss History
plt.subplot(131)
plt.plot(model.loss_history)
plt.title('Loss History')
plt.xlabel('Iteration')
plt.ylabel('MSE Loss')

# Plot 2: Predicted vs Actual (Training)
plt.subplot(132)
plt.scatter(y_train, y_train_pred, alpha=0.5)
plt.plot([-10, 10], [-10, 10], 'r--')  # Perfect prediction line
plt.title('Predicted vs Actual (Training)')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

# Plot 3: Predicted vs Actual (Testing)
plt.subplot(133)
plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.plot([-10, 10], [-10, 10], 'r--')  # Perfect prediction line
plt.title('Predicted vs Actual (Testing)')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

plt.tight_layout()
plt.show()

