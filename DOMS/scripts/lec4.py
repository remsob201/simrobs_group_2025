""" Particle Swarm Optimization (PSO) for Rastrigin Function
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# --- Rastrigin Function ---
def rastrigin(X, A=10):
    return A * X.shape[1] + np.sum(X**2 - A * np.cos(2 * np.pi * X), axis=1)


# Parameters
NUM_PARTICLES = 50
NUM_DIMENSIONS = 2
MAX_ITER = 100
BOUNDS = np.array([[-5.12, 5.12], [-5.12, 5.12]]) # Same for both dims

# PSO parameters
w = 0.7 # inertia
c1 = 1.5 # cognitive
c2 = 1.5 # social

# Initialize positions and velocities
positions = np.random.uniform(low=BOUNDS[:,
                                         0],
                              high=BOUNDS[:,
                                          1],
                              size=(NUM_PARTICLES,
                                    NUM_DIMENSIONS))
velocities = np.random.randn(NUM_PARTICLES, NUM_DIMENSIONS)

# Initialize personal and global bests
personal_best_pos = positions.copy()
personal_best_val = rastrigin(positions)
global_best_pos = personal_best_pos[np.argmin(personal_best_val)]

# Create meshgrid for visualization
X1, X2 = np.meshgrid(np.linspace(*BOUNDS[0], 200), np.linspace(*BOUNDS[1], 200))
Z = rastrigin(np.stack([X1.ravel(), X2.ravel()], axis=1)).reshape(X1.shape)

# 3D Visualization setup
plt.ion()
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# --- PSO Main Loop ---
for iteration in range(MAX_ITER):
    fitness = rastrigin(positions)

    # Update personal and global bests
    better_mask = fitness < personal_best_val
    personal_best_val[better_mask] = fitness[better_mask]
    personal_best_pos[better_mask] = positions[better_mask]

    best_idx = np.argmin(personal_best_val)
    if personal_best_val[best_idx] < rastrigin(global_best_pos.reshape(1, -1)):
        global_best_pos = personal_best_pos[best_idx].copy()

    # Update velocity and position
    r1, r2 = np.random.rand(2, NUM_PARTICLES, NUM_DIMENSIONS)
    velocities = (w * velocities + c1 * r1 * (personal_best_pos - positions) + c2 * r2 *
                  (global_best_pos - positions))
    positions += velocities

    # Clamp within bounds
    for dim in range(NUM_DIMENSIONS):
        positions[:, dim] = np.clip(positions[:, dim], BOUNDS[dim, 0], BOUNDS[dim, 1])

    # Visualization
    ax.clear()
    ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.6, rstride=10, cstride=10, edgecolor='none')
    ax.scatter(positions[:,
                         0],
               positions[:,
                         1],
               rastrigin(positions),
               color='white',
               edgecolor='black',
               label='Particles')
    ax.scatter(global_best_pos[0],
               global_best_pos[1],
               rastrigin(global_best_pos.reshape(1,
                                                 -1)),
               color='red',
               s=80,
               label='Global Best')
    ax.set_title(f"Iteration {iteration+1}")
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_zlabel('Rastrigin Value')
    ax.view_init(elev=45, azim=45)
    ax.legend()
    plt.pause(0.2)

plt.ioff()
plt.show()
