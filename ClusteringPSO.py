import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import time

# Generate synthetic 3D dataset with points randomly distributed in [-7, 7]
def generate_dataset(n_points=300):
    points = []
    for _ in range(n_points):
        # Assign points uniformly in [-7, 7] for each dimension
        points.append([random.uniform(-7, 7), random.uniform(-7, 7), random.uniform(-7, 7)])
    return np.array(points)

# Compute Euclidean distance between two 3D points
def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)

# Objective function: Sum of squared distances from points to nearest centroid
def compute_fitness(centroids, data):
    total_distance = 0
    for point in data:
        distances = [distance(point, centroid) for centroid in centroids]
        total_distance += min(distances) ** 2
    return total_distance

# Assign points to clusters and return assignments
def assign_clusters(centroids, data):
    assignments = []
    for point in data:
        distances = [distance(point, centroid) for centroid in centroids]
        cluster_idx = np.argmin(distances)
        assignments.append(cluster_idx)
    return np.array(assignments)

# Update centroids to the mean of their assigned clusters
def update_centroids(centroids, data, assignments, n_clusters):
    new_centroids = []
    for i in range(n_clusters):
        cluster_points = data[assignments == i]
        if len(cluster_points) > 0:
            new_centroids.append(np.mean(cluster_points, axis=0))
        else:
            new_centroids.append(centroids[i])  # Keep old centroid if no points assigned
    return np.array(new_centroids)

# Particle class representing a solution (set of 3D centroids)
class Particle:
    def __init__(self, centroids, n_clusters):
        self.centroids = np.array(centroids)  # Shape: (n_clusters, 3)
        self.velocity = np.random.uniform(-1, 1, (n_clusters, 3))  # Random initial velocity
        self.best_centroids = self.centroids.copy()
        self.fitness = float('inf')
        self.best_fitness = float('inf')

# Initialize population with centroids in plot's axis range [-7, 7]
def initialize_population():
    particles = []
    for _ in range(n_particles):
        # Initialize all centroids randomly in [-7, 7]
        centroids = [[random.uniform(-7, 7), random.uniform(-7, 7), random.uniform(-7, 7)]
                     for _ in range(n_clusters)]
        particles.append(Particle(centroids, n_clusters))
    return particles

# PSO parameters
n_particles = 10  # Reduced to focus optimization
n_clusters = 3
max_iterations = 20  # Set to achieve convergence by 20 iterations
w = 0.8  # Higher inertia weight for initial exploration
c1 = 1.0  # Reduced cognitive coefficient for slower convergence
c2 = 1.0  # Reduced social coefficient for slower convergence
max_velocity = 5.0  # Maximum velocity for centroid movement

# Setup
data = generate_dataset()
particles = initialize_population()
global_best_fitness = float('inf')
global_best_centroids = None

# Visualization setup
plt.ion()  # Interactive mode for dynamic updates
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Main PSO loop
iteration = 0
prev_global_best_centroids = None
centroid_movement_threshold = 1e-4  # Small threshold for centroid movement

while iteration < max_iterations:
    # Evaluate fitness
    for particle in particles:
        particle.fitness = compute_fitness(particle.centroids, data)
        # Update personal best
        if particle.fitness < particle.best_fitness:
            particle.best_fitness = particle.fitness
            particle.best_centroids = particle.centroids.copy()
        # Update global best
        if particle.fitness < global_best_fitness:
            global_best_fitness = particle.fitness
            global_best_centroids = particle.centroids.copy()

    # Check if centroids have stopped moving
    if prev_global_best_centroids is not None:
        max_movement = np.max([distance(prev_global_best_centroids[i], global_best_centroids[i])
                              for i in range(n_clusters)])
        if max_movement < centroid_movement_threshold:
            print(f"Centroids stopped moving (max movement: {max_movement:.6f}). Stopping at iteration {iteration}.")
            break

    # Store current global best centroids for next iteration
    prev_global_best_centroids = global_best_centroids.copy()

    # Update velocities and positions
    for particle in particles:
        r1 = random.random()
        r2 = random.random()
        # Update velocity
        particle.velocity = (w * particle.velocity +
                            c1 * r1 * (particle.best_centroids - particle.centroids) +
                            c2 * r2 * (global_best_centroids - particle.centroids))
        # Clip velocity to prevent excessive movement
        particle.velocity = np.clip(particle.velocity, -max_velocity, max_velocity)
        # Update position
        particle.centroids = particle.centroids + particle.velocity
        particle.centroids = np.clip(particle.centroids, -7, 7)
        # Recenter centroids to cluster means
        assignments = assign_clusters(particle.centroids, data)
        particle.centroids = update_centroids(particle.centroids, data, assignments, n_clusters)

    # Gradually reduce inertia weight
    w = w * 0.98  # Slower decay for balanced exploration/exploitation
    iteration += 1

    # Visualization
    ax.clear()
    assignments = assign_clusters(global_best_centroids, data)
    cluster_colors = ['r', 'b', 'g']
    for i in range(n_clusters):
        cluster_points = data[assignments == i]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2],
                   c=cluster_colors[i], s=30, alpha=0.5, label=f'Cluster {i + 1}')

    # Plot best centroids
    ax.scatter(global_best_centroids[:, 0], global_best_centroids[:, 1], global_best_centroids[:, 2],
               c=cluster_colors, s=200, marker='*', edgecolors='k', label='Centroids')

    # Plot particles
    for particle in particles:
        ax.scatter(particle.centroids[:, 0], particle.centroids[:, 1], particle.centroids[:, 2],
                   c='purple', s=50, marker='o', alpha=0.3)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-7, 7)
    ax.set_ylim(-7, 7)
    ax.set_zlim(-7, 7)
    ax.set_title(f'PSO Clustering (Iteration {iteration})')
    ax.legend()
    plt.draw()
    plt.pause(1.0)  # Pause for slower, more visible movements

plt.ioff()
plt.show()
