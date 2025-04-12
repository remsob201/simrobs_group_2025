"""Pattern Search Optimization with Visualization
This code implements a simple pattern search optimization algorithm to find the minimum of a given function.
It visualizes the optimization process in 2D.
"""

import numpy as np
import matplotlib.pyplot as plt


# Define the objective function
def objective(x):
    return np.sin(2 * x) + 3 * np.cos(0.2 * x)


# Pattern search parameters
x0 = np.random.uniform(0, 10) # initial guess
# x0 = 0.0 # fixed initial guess for reproducibility
step_size = 3.0
tol = 1e-2
max_iter = 100
alpha = .5 # step size reduction factor
FULL_POLL = False # set to False for greedy (first improving direction)

# Plot setup
x_plot = np.linspace(0, 10, 1000)
y_plot = objective(x_plot)

plt.ion()
fig, ax = plt.subplots(figsize=(10, 6))

x_curr = x0
for iteration in range(max_iter):
    ax.clear()
    y_curr = objective(x_curr)

    improved = False
    trial_points = []

    if FULL_POLL:
        # Evaluate both directions and pick best
        directions = [-1, 1]
        for d in directions:
            x_trial = np.clip(x_curr + d * step_size, 0, 10)
            trial_points.append(x_trial)

        y_trials = [objective(x) for x in trial_points]
        all_x = [x_curr] + trial_points
        all_y = [y_curr] + y_trials

        best_idx = np.argmin(all_y)
        x_next = all_x[best_idx]

        if best_idx != 0:
            improved = True
    else:
        # Greedy: move to first improving direction
        for d in [-1, 1]:
            x_trial = np.clip(x_curr + d * step_size, 0, 10)
            y_trial = objective(x_trial)
            trial_points.append(x_trial)
            if y_trial < y_curr:
                x_next = x_trial
                improved = True
                break
        if not improved:
            x_next = x_curr

    # Plot current step
    ax.plot(x_plot, y_plot, label="F(x)", linewidth=2)
    ax.scatter(trial_points, [objective(x) for x in trial_points], c='blue', label="Trial Points")
    ax.plot(x_next, objective(x_next), 'ro', label="Current Best")
    ax.set_title(f"Iteration {iteration + 1}")
    ax.set_xlabel("x")
    ax.set_ylabel("F(x)")
    ax.set_xlim(0, 10)
    ax.set_ylim(min(y_plot) - 1, max(y_plot) + 1)
    ax.grid(True)
    ax.legend()
    plt.pause(0.5)

    if not improved:
        step_size *= alpha
    x_curr = x_next

    if step_size < tol:
        break

plt.ioff()
plt.show()

# Final result
final_y = objective(x_curr)
print(f"\nBest x found: {x_curr:.5f}")
print(f"Minimum value F(x): {final_y:.5f}")
