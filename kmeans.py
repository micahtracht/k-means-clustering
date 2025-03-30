import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def k_means(vectors, k, max_iter=1000, tol=0.01):
    """
    Performs k-means clustering on a list of n-dimensional vectors.
    
    Parameters:
        vectors (list): List of vectors
        k (int): Number of clusters.
        max_iter (int): Max number of iterations.
        tol (float): Tolerance for convergence (algorithm stops when the change in cost is below tol).
    
    Returns:
        cost (float): Final cost (avg squared distance).
        assignments (list): List of [vector, cluster_index] pairs.
        centroids (list): List of k centroid vectors.
        costs (list): The cost (avg squared distance) at each iteration.
        history (list): (assignments, centroids) at each iteration
    """
    # Helper to compute squared Euclidean distance
    def dist_squared(u, v):
        return sum((u[i] - v[i])**2 for i in range(len(u)))
    
    # Compute cost as the avg squared distance of points to their assigned centroid (euclidean)
    def j_clust(assignments, centroids):
        total = 0.0
        for vec, group in assignments:
            total += dist_squared(vec, centroids[group])
        return total / len(assignments)
    
    # Assign each vector to nearest centroid
    def assign_groups(assignments, centroids):
        for i, (vec, _) in enumerate(assignments):
            best_group = 0
            min_dist = float('inf')
            for j in range(k):
                d = dist_squared(vec, centroids[j])
                if d < min_dist:
                    min_dist = d
                    best_group = j
            assignments[i][1] = best_group
    
    # Update centroids to be the mean of all vectors assigned to each cluster
    def assign_centroids(assignments, centroids):
        groups = {i: [] for i in range(k)}
        for vec, group in assignments:
            groups[group].append(vec)
        for j in range(k):
            if groups[j]:  # if the cluster is not empty, compute its mean
                new_centroid = [sum(coords)/len(coords) for coords in zip(*groups[j])]
                centroids[j] = new_centroid
            else:
                # If no vectors are assigned to this cluster, reinitialize its centroid randomly.
                centroids[j] = list(np.random.rand(len(vectors[0])))
    
    # Initialize assignments (each vector is paired with 0 as a default cluster)
    assignments = [[vec, 0] for vec in vectors]
    
    # Initialize centroids with random vectors
    centroids = [list(np.random.rand(len(vectors[0]))) for _ in range(k)]
    
    prev_cost = float('inf')
    cost = j_clust(assignments, centroids)
    num_iter = 0
    
    # Iterate until convergence or maximum iterations reached.
    costs = []
    history = []
    while abs(prev_cost - cost) > tol and num_iter < max_iter:
        num_iter += 1
        assign_groups(assignments, centroids)
        assign_centroids(assignments, centroids)
        prev_cost = cost
        cost = j_clust(assignments, centroids)
        costs.append(cost)
        
        assignment_snapshot = [list(a) for a in assignments]
        centroid_snapshot = [list(c) for c in centroids]
        history.append((assignment_snapshot, centroid_snapshot))
    
    return cost, assignments, centroids, costs, history

vectors = [[np.random.rand() * np.random.randint(-10, 11), np.random.rand() * np.random.randint(-10, 11)] for _ in range(1000)]
cost, assignments, centroids, costs, history = k_means(vectors, k=7, max_iter=1000, tol=0.001)
print(costs)
# do the plotting
def plot_progression(costs):
    plt.plot(range(len(costs)), costs, marker='o')
    plt.title('K-Means convergence: Cost vs iterations')
    plt.xlabel('number of iterations')
    plt.ylabel('Cost (avg squared distance)')
    plt.grid(True)
    plt.show()

def plot_result(assignments, centroids):
    X_vals = [vec[0] for vec, _ in assignments]
    Y_vals = [vec[1] for vec, _ in assignments]
    colors = [group for _, group in assignments]
    
    plt.scatter(X_vals, Y_vals, c=colors, cmap='tab10', alpha=0.7)
    centroid_X = [c[0] for c in centroids]
    centroid_Y = [c[1] for c in centroids]
    
    plt.scatter(centroid_X, centroid_Y, marker = 'X', c='red', label='Centroids/representatives')
    
    plt.title('K-Means clustering (2D only)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()

def show_kmeans_animation(history):
    fig, ax = plt.subplots()

    def update(frame):
        ax.clear()
        assignments, centroids = history[frame]
        vectors = [vec for vec, _ in assignments]
        labels = [group for _, group in assignments]
        vectors = np.array(vectors)
        centroids = np.array(centroids)

        ax.scatter(vectors[:, 0], vectors[:, 1], c=labels, cmap='tab10', alpha=0.7)
        ax.scatter(centroids[:, 0], centroids[:, 1], marker='X', c='black', s=100)
        ax.set_title(f"Iteration {frame + 1}")

    ani = FuncAnimation(fig, update, frames=len(history), interval=800, repeat=False)
    plt.show()

def elbow_method_plot(max_k):
    cost_vals = []
    for i in range(1, max_k):
        cost_vals.append(k_means(vectors, k=i)[0])
    
    plt.plot(range(len(cost_vals)), cost_vals, marker = 'o')
    plt.title('Cost vs centroid # elbow graph')
    plt.xlabel('Number of centroids')
    plt.ylabel('avg squared distance')
    plt.grid(True)
    plt.show()

elbow_method_plot(100)