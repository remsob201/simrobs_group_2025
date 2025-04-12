"""
# Genetic Algorithm for Rastrigin Function
This code implements a genetic algorithm to minimize the Rastrigin function, a well-known benchmark function in optimization.
It visualizes the optimization process in 3D.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# --- Rastrigin Function ---
def rastrigin(X, A=10):
    return A * X.shape[1] + np.sum(X**2 - A * np.cos(2 * np.pi * X), axis=1)


# Parameters
POP_SIZE = 50
NUM_DIMENSIONS = 2
MAX_GENERATIONS = 100
BOUNDS = np.array([[-5.12, 5.12], [-5.12, 5.12]])
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.9
TOURNAMENT_SIZE = 3


# --- Genetic Algorithm Functions ---
def initialize_population():
    return np.random.uniform(BOUNDS[:, 0], BOUNDS[:, 1], size=(POP_SIZE, NUM_DIMENSIONS))


def evaluate_fitness(pop):
    return rastrigin(pop)


def tournament_selection(pop, fitness):
    selected = []
    for _ in range(POP_SIZE):
        idx = np.random.choice(range(POP_SIZE), TOURNAMENT_SIZE, replace=False)
        winner = idx[np.argmin(fitness[idx])]
        selected.append(pop[winner])
    return np.array(selected)


def crossover(parent1, parent2):
    if np.random.rand() < CROSSOVER_RATE:
        alpha = np.random.rand()
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = alpha * parent2 + (1 - alpha) * parent1
        return child1, child2
    else:
        return parent1.copy(), parent2.copy()


def mutate(individual):
    for i in range(NUM_DIMENSIONS):
        if np.random.rand() < MUTATION_RATE:
            individual[i] += np.random.normal(0, 0.5)
            individual[i] = np.clip(individual[i], BOUNDS[i, 0], BOUNDS[i, 1])
    return individual


# --- Create meshgrid for 3D plot ---
X1, X2 = np.meshgrid(np.linspace(*BOUNDS[0], 200), np.linspace(*BOUNDS[1], 200))
Z = rastrigin(np.stack([X1.ravel(), X2.ravel()], axis=1)).reshape(X1.shape)

# --- Initialize population ---
population = initialize_population()
best_fitness = float('inf')
best_individual = None

# 3D Visualization setup
plt.ion()
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# --- Genetic Algorithm Loop ---
for generation in range(MAX_GENERATIONS):
    fitness = evaluate_fitness(population)
    gen_best_idx = np.argmin(fitness)

    if fitness[gen_best_idx] < best_fitness:
        best_fitness = fitness[gen_best_idx]
        best_individual = population[gen_best_idx].copy()

    # Selection
    selected = tournament_selection(population, fitness)

    # Crossover and Mutation
    next_population = []
    for i in range(0, POP_SIZE, 2):
        parent1 = selected[i]
        parent2 = selected[(i + 1) % POP_SIZE]
        child1, child2 = crossover(parent1, parent2)
        next_population.append(mutate(child1))
        next_population.append(mutate(child2))

    population = np.array(next_population)

    # --- Visualization ---
    ax.clear()
    ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.6, rstride=10, cstride=10, edgecolor='none')
    ax.scatter(population[:,
                          0],
               population[:,
                          1],
               rastrigin(population),
               color='white',
               edgecolor='black',
               label='Population')
    ax.scatter(best_individual[0],
               best_individual[1],
               rastrigin(best_individual.reshape(1,
                                                 -1)),
               color='red',
               s=80,
               label='Best')
    ax.set_title(f"Generation {generation+1}")
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_zlabel('Rastrigin Value')
    ax.view_init(elev=45, azim=45)
    ax.legend()
    plt.pause(0.2)

plt.ioff()
plt.show()

print(f"Best solution found: {best_individual}, Fitness: {best_fitness:.4f}")
