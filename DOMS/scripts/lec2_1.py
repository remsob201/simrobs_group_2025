"""
# Genetic Algorithm for Function Optimization
This code implements a simple genetic algorithm to find the minimum of a given function.
It uses a population of solutions, selection, crossover, and mutation to evolve the population over generations.
"""
import numpy as np
import matplotlib.pyplot as plt


# Define the objective function
def objective(x):
    return np.sin(x * 2) + 3 * np.cos(0.2 * x)


# Genetic Algorithm Parameters
POP_SIZE = 1000
N_GENERATIONS = 50
X_BOUND = [0, 10]
MUTATION_RATE = 0.1
ELITISM = True

# Initialize population
population = np.random.uniform(X_BOUND[0], X_BOUND[1], size=POP_SIZE)


# Evaluate fitness
def fitness(x):
    return -objective(x)


# Store best individuals for visualization
best_history = []

# Set up plot
x_plot = np.linspace(X_BOUND[0], X_BOUND[1], 1000)
y_plot = objective(x_plot)

plt.ion() # Turn on interactive mode
fig, ax = plt.subplots(figsize=(10, 6))

for gen in range(N_GENERATIONS):
    ax.clear()

    # Selection (Tournament)
    idx = np.random.randint(0, POP_SIZE, size=(POP_SIZE, 2))
    selected = np.where(
        fitness(population[idx[:,
                               0]]) > fitness(population[idx[:,
                                                             1]]),
        population[idx[:,
                       0]],
        population[idx[:,
                       1]])

    # Crossover (Arithmetic)
    alpha = np.random.rand(POP_SIZE)
    parent1 = selected
    parent2 = np.roll(selected, 1)
    offspring = alpha * parent1 + (1 - alpha) * parent2

    # Mutation
    mutation = np.random.rand(POP_SIZE) < MUTATION_RATE
    offspring[mutation] += np.random.normal(0, 0.1, np.sum(mutation))
    offspring = np.clip(offspring, X_BOUND[0], X_BOUND[1])

    # Elitism
    if ELITISM:
        best_idx = np.argmax(fitness(population))
        worst_idx = np.argmin(fitness(offspring))
        offspring[worst_idx] = population[best_idx]

    population = offspring

    # Save best of generation
    best_x = population[np.argmax(fitness(population))]
    best_y = objective(best_x)
    best_history.append((best_x, best_y))

    # Plot function
    ax.plot(x_plot, y_plot, label="F(x)", linewidth=2)

    # Plot current population
    y_pop = objective(population)
    ax.scatter(population, y_pop, c='blue', s=10, label='Population')

    # Plot best individual this generation
    ax.plot(best_x, best_y, 'ro', label='Best of Generation')

    ax.set_title(f"Generation {gen+1}")
    ax.set_xlabel("x")
    ax.set_ylabel("F(x)")
    ax.set_xlim(X_BOUND[0], X_BOUND[1])
    ax.set_ylim(min(y_plot) - 1, max(y_plot) + 1)
    ax.legend()
    ax.grid(True)

    plt.pause(0.1)

plt.ioff() # Turn off interactive mode

# Final best result
final_best_x, final_best_y = min(best_history, key=lambda xy: xy[1])
print(f"\nFinal best x: {final_best_x:.4f}")
print(f"Final min F(x): {final_best_y:.4f}")

# Plot full result with history
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(x_plot, y_plot, label="F(x)", linewidth=2)
all_best_x, all_best_y = zip(*best_history)
ax2.plot(all_best_x, all_best_y, 'o-', color='orange', label='Best Each Generation')
ax2.plot(final_best_x, final_best_y, 'ro', label='Final Best')
ax2.set_title("Optimization Summary")
ax2.set_xlabel("x")
ax2.set_ylabel("F(x)")
ax2.legend()
ax2.grid(True)
plt.show()
