"""Pattern Search Optimization with Rastrigin Function Visualization"
This code implements a pattern search optimization algorithm to minimize the Rastrigin function, a well-known benchmark function in optimization. The algorithm iteratively explores the search space and visualizes the optimization process in 3D.
The Rastrigin function is defined as:
f(x) = A * n + sum(x_i^2 - A * cos(2 * pi * x_i))
where A is a constant (typically 10), n is the number of dimensions, and x_i are the individual components of the input vector x.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# --- Rastrigin Function ---
def rastrigin(X, A=10):
    return A * X.shape[1] + np.sum(X**2 - A * np.cos(2 * np.pi * X), axis=1)


# Parameters
NUM_DIMENSIONS = 2
MAX_ITERATIONS = 100
INITIAL_STEP_SIZE = 0.5
CONVERGENCE_THRESHOLD = 1e-6
BOUNDS = np.array([[-5.12, 5.12], [-5.12, 5.12]])


# --- Pattern Search Optimization Function ---
def pattern_search(initial_guess):
    best_solution = initial_guess.copy()
    best_fitness = rastrigin(best_solution.reshape(1, -1))

    current_solution = best_solution.copy()
    current_fitness = best_fitness

    step_size = INITIAL_STEP_SIZE
    iteration = 0

    # --- Create meshgrid for 3D plot ---
    X1, X2 = np.meshgrid(np.linspace(*BOUNDS[0], 200), np.linspace(*BOUNDS[1], 200))
    Z = rastrigin(np.stack([X1.ravel(), X2.ravel()], axis=1)).reshape(X1.shape)

    # 3D Visualization setup
    plt.ion()
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    while iteration < MAX_ITERATIONS:
        # Try all directions in the search space
        directions = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]]) # Right, Up, Left, Down directions
        new_solution_found = False

        for direction in directions:
            # Calculate the new candidate solution
            candidate_solution = current_solution + step_size * direction
            # Clip the solution to respect bounds
            candidate_solution = np.clip(candidate_solution, BOUNDS[:, 0], BOUNDS[:, 1])

            candidate_fitness = rastrigin(candidate_solution.reshape(1, -1))

            # If better solution is found, update the current solution
            if candidate_fitness < current_fitness:
                current_solution = candidate_solution
                current_fitness = candidate_fitness
                new_solution_found = True
                break

        # If no better solution was found, reduce step size and continue
        if not new_solution_found:
            step_size /= 2.0 # Decrease step size

        # Track the best solution found so far
        if current_fitness < best_fitness:
            best_solution = current_solution.copy()
            best_fitness = current_fitness

        # --- Visualization ---
        ax.clear()
        ax.plot_surface(X1,
                        X2,
                        Z,
                        cmap='viridis',
                        alpha=0.6,
                        rstride=10,
                        cstride=10,
                        edgecolor='none')
        ax.scatter(current_solution[0],
                   current_solution[1],
                   rastrigin(current_solution.reshape(1,
                                                      -1)),
                   color='white',
                   edgecolor='black',
                   label='Current Solution')
        ax.scatter(best_solution[0],
                   best_solution[1],
                   rastrigin(best_solution.reshape(1,
                                                   -1)),
                   color='red',
                   s=80,
                   label='Best Solution')
        ax.set_title(f"Iteration {iteration+1}")
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        ax.set_zlabel('Rastrigin Value')
        ax.view_init(elev=45, azim=45)
        ax.legend()
        plt.pause(0.2)

        # Check convergence
        if step_size < CONVERGENCE_THRESHOLD:
            break

        iteration += 1

    plt.ioff()
    plt.show()

    return best_solution, best_fitness


# Initial guess for Pattern Search
initial_guess = np.random.uniform(BOUNDS[:, 0], BOUNDS[:, 1], NUM_DIMENSIONS)

# Run Pattern Search
best_solution, best_fitness = pattern_search(initial_guess)

print(f"Best solution found: {best_solution}, Fitness: {best_fitness[0]:.4f}")
