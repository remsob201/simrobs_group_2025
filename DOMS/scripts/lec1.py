"""This script demonstrates three optimization methods: Brute Force Search, Gradient Descent, and Simulated Annealing.
It uses a sample function to find the minimum value and visualize the results.
"""

import numpy as np
import matplotlib.pyplot as plt


# Brute Force Search
def brute_force(values):
    min_value = values[0]
    iterations = 0
    for val in values:
        iterations += 1
        if val < min_value:
            min_value = val
    return min_value, iterations


# Gradient Descent with Visualization
def gradient_descent(x, function, learning_rate=0.01, max_iterations=1000, epsilon=0.0001):
    # x_current = np.random.choice(x) # Initial random guess
    x_current = 8.0
    initial_x = x_current # Store initial guess
    iterations = 0

    for _ in range(max_iterations):
        iterations += 1
        grad = (np.interp(x_current + 1e-5, x, function) - np.interp(x_current, x, function)) / 1e-5
        x_new = x_current - learning_rate * grad

        if abs(np.interp(x_new, x, function) - np.interp(x_current, x, function)) < epsilon:
            break

        x_current = np.clip(x_new, x[0], x[-1])

    return initial_x, x_current, np.interp(x_current, x, function), iterations


# Simulated Annealing with Visualization
def simulated_annealing(x,
                        function,
                        initial_temp=100,
                        cooling_rate=0.99,
                        max_iterations=1000,
                        threshold=3):
    x_current = np.random.choice(x) # Initial random guess
    # x_current = 7.8 # Initial manual guess
    initial_x = x_current # Store initial guess
    f_current = np.interp(x_current, x, function)
    best_x, best_f = x_current, f_current
    temp = initial_temp
    iterations = 0
    no_improve_count = 0

    for _ in range(max_iterations):
        iterations += 1
        x_new = x_current + np.random.uniform(-1, 1)
        x_new = np.clip(x_new, x[0], x[-1])
        f_new = np.interp(x_new, x, function)

        if f_new < f_current or np.exp((f_current - f_new) / temp) > np.random.rand():
            x_current, f_current = x_new, f_new
            no_improve_count = 0
        else:
            no_improve_count += 1

        if f_current < best_f:
            best_x, best_f = x_current, f_current

        temp *= cooling_rate

        if no_improve_count > threshold:
            break

    return initial_x, best_x, best_f, iterations


# Define function
x = np.arange(0, 10., 0.01)
F = np.sin(x * 2) + 5 * np.sin(x) * np.cos(0.2 * x)**2

# Compute minima using different methods
min_value_brute, iters_brute = brute_force(F)
initial_x_gd, min_x_gd, min_value_gd, iters_gd = gradient_descent(x, F)
initial_x_sa, min_x_sa, min_value_sa, iters_sa = simulated_annealing(x, F)

# Plot results
plt.plot(x, F, label="Function", linewidth=2)
plt.axhline(min_value_brute, color='r', linestyle='--', label=f"Brute Force: {min_value_brute:.2f}")

# Plot initial guesses
plt.scatter(x[0], F[0], color='red', marker='o', label="Brute Force Start", zorder=3)
plt.scatter(initial_x_gd,
            np.interp(initial_x_gd,
                      x,
                      F),
            color='green',
            marker='o',
            label="GD Start",
            zorder=3)
plt.scatter(initial_x_sa,
            np.interp(initial_x_sa,
                      x,
                      F),
            color='blue',
            marker='o',
            label="SA Start",
            zorder=3)

# Plot final solutions
plt.scatter(min_x_gd,
            min_value_gd,
            color='green',
            marker='x',
            label=f"GD: {min_value_gd:.2f} ({iters_gd} iter)",
            zorder=3)
plt.scatter(min_x_sa,
            min_value_sa,
            color='blue',
            marker='x',
            label=f"SA: {min_value_sa:.2f} ({iters_sa} iter)",
            zorder=3)

plt.legend()
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Function Optimization Methods")
plt.show()

# Print results
print(f"Brute Force Min: {min_value_brute:.5f} (Iterations: {iters_brute})")
print(f"Gradient Descent Min: x={min_x_gd:.5f}, f(x)={min_value_gd:.5f} (Iterations: {iters_gd})")
print(
    f"Simulated Annealing Min: x={min_x_sa:.5f}, f(x)={min_value_sa:.5f} (Iterations: {iters_sa})")
# Note: The code above includes the brute force search, gradient descent, and simulated annealing methods
