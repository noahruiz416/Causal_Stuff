import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def StochasticGradientDescent(data, learning_rate, iter):
    """

    input:
            1. data (vector of data to estimate parameters from),
            2. learning_rate (step of an iteration in gradient descent)
            3. iterations (number of times algorithm will be ran)

    output:
            1. Parameters (Theta_0, Theta_1)

    """
    num_features = data.shape[1] - 1  # Calculate the number of features
    theta = np.zeros(num_features + 1)  # Initialize parameters, including an intercept term
    alpha = learning_rate
    n = data.shape[0]

    for _ in range(iter):
        for i in range(n):
            x_i = data[i, :-1]  # Extract features (all but the last column)
            y_i = data[i, -1]  # Extract the target variable (last column)
            error = y_i - np.dot(theta, np.insert(x_i, 0, 1))  # Include an intercept term
            gradient = -2 * error * np.insert(x_i, 0, 1)
            theta -= alpha * gradient

    return theta

#we can now test our algo against simulated data.

# Define the parameters of the linear relationship
m = 2  # Slope
b = 1  # Intercept

# Generate a range of x values
x = np.linspace(0, 10, 100)

# Calculate the corresponding y values without noise
y_true = m * x + b

# Add random Gaussian noise to y
mean = 0  # Mean of the Gaussian noise
stddev = 1  # Standard deviation of the Gaussian noise
noise = np.random.normal(mean, stddev, len(x))
y_with_noise = y_true + noise

data = np.column_stack((x, y_with_noise))

theta = LinReg(data, 0.01, 1000)

print(theta)

y_pred = theta[0] + (theta[1] * x)


# Plot the data
# Plot the data
plt.figure(figsize=(8, 6))
plt.scatter(x, y_with_noise, label='Data with Noise', color='blue')
plt.plot(x, y_true, label='True Relationship', color='red', linewidth=2)
plt.plot(x, y_pred, label='Model Predictions', color='green', linestyle='--')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Linear Relationship with Hand Made Model')
plt.grid(True)
plt.show()
