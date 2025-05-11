import numpy as np

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def manhattan_distance(a, b):
    return np.sum(np.abs(a - b))

def k_means(X, k, max_iters=100, random_state=None):
    # Randomly initialize centroids
    rng = np.random.default_rng(random_state)
    centroids = X[rng.choice(X.shape[0], k, replace=False)]
    for _ in range(max_iters):
        # Assign points to the nearest centroid
        labels = np.array(
            [np.argmin([euclidean_distance(x, c) for c in centroids]) for x in X]
        )
        # Update centroids as the mean of assigned points
        new_centroids = np.array(
            [
                X[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
                for i in range(k)
            ]
        )
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return labels, centroids

def k_medians(X, k, max_iters=100, random_state=None):
    # Randomly initialize medians
    rng = np.random.default_rng(random_state)
    medians = X[rng.choice(X.shape[0], k, replace=False)]
    for _ in range(max_iters):
        # Assign points to the nearest median
        labels = np.array(
            [np.argmin([manhattan_distance(x, m) for m in medians]) for x in X]
        )
        # Update medians as the median of assigned points
        new_medians = np.array(
            [
                np.median(X[labels == i], axis=0) if np.any(labels == i) else medians[i]
                for i in range(k)
            ]
        )
        if np.all(medians == new_medians):
            break
        medians = new_medians
    return labels, medians

def cluster_linkage_distance(X, c1, c2, linkage="single"):
        n = len(X)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                distances[i, j] = euclidean_distance(X[i], X[j])
                distances[j, i] = distances[i, j]
        # Compute the distance between two clusters according to the linkage method
        points1 = c1
        points2 = c2
        dists = [distances[i, j] for i in points1 for j in points2]
        if linkage == "single":
            # Single linkage: minimum distance
            return np.min(dists)
        elif linkage == "complete":
            # Complete linkage: maximum distance
            return np.max(dists)
        elif linkage == "average":
            # Average linkage: mean distance
            return np.mean(dists)
        elif linkage == "centroid":
            # Centroid linkage: distance between cluster centroids (means)
            X1 = X[points1]
            X2 = X[points2]
            mean1 = np.mean(X1, axis=0)
            mean2 = np.mean(X2, axis=0)
            return euclidean_distance(mean1, mean2)
        elif linkage == "median":
            # Median linkage: distance between cluster medians
            X1 = X[points1]
            X2 = X[points2]
            median1 = np.median(X1, axis=0)
            median2 = np.median(X2, axis=0)
            return euclidean_distance(median1, median2)
        elif linkage == "ward":
            # Ward's method: increase in within-cluster variance when merging clusters
            X1 = X[points1]
            X2 = X[points2]
            n1 = len(X1)
            n2 = len(X2)
            mean1 = np.mean(X1, axis=0)
            mean2 = np.mean(X2, axis=0)
            mean_all = (n1 * mean1 + n2 * mean2) / (n1 + n2)
            ss1 = np.sum((X1 - mean1) ** 2)
            ss2 = np.sum((X2 - mean2) ** 2)
            ss_all = np.sum((np.vstack((X1, X2)) - mean_all) ** 2)
            return ss_all - ss1 - ss2
        else:
            raise ValueError(
                f"Unknown linkage: {linkage}."
            )

def hierarchical_clustering(X, linkage="single"):
    n = len(X)
    clusters = [[i] for i in range(n)]
    res = []
    cluster_ids = list(range(n))
    next_cluster_id = n

    while len(clusters) > 1:
        min_dist = float("inf")
        to_merge = (0, 0)
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                dist = cluster_linkage_distance(X, clusters[i], clusters[j], linkage)
                if dist < min_dist:
                    min_dist = dist
                    to_merge = (i, j)
        i, j = to_merge

        # Save the linkage matrix row: [idx1, idx2, dist, sample_count]
        idx1 = cluster_ids[i]
        idx2 = cluster_ids[j]
        merged_cluster = clusters[i] + clusters[j]
        res.append([idx1, idx2, min_dist, len(merged_cluster)])
        
        # Merge clusters
        clusters[i] = merged_cluster
        del clusters[j]
        cluster_ids[i] = next_cluster_id
        del cluster_ids[j]
        next_cluster_id += 1

    return np.array(res, dtype=float)
    

def nn_clustering(X, threshold):
    n = len(X)
    visited = [False] * n
    clusters = []

    for i in range(n):
        if visited[i]:
            continue

        # Start a new cluster
        cluster = [i]
        visited[i] = True
        to_process = [i]

        while to_process:
            current = to_process.pop()
            for j in range(n):
                if not visited[j]:
                    dist = euclidean_distance(X[current], X[j])
                    if dist <= threshold:
                        visited[j] = True
                        cluster.append(j)
                        to_process.append(j)

        clusters.append(cluster)

    return clusters

def region_query(X, point_idx, eps):
        # Find all points within eps distance of a given point
        return [
            i for i in range(len(X)) if euclidean_distance(X[point_idx], X[i]) <= eps
        ]

def expand_cluster(X, labels, point_idx, neighbors, cluster_id, eps, min_samples):
    # Assign the cluster id to the starting point
    labels[point_idx] = cluster_id
    i = 0
    while i < len(neighbors):
        neighbor_idx = neighbors[i]
        if labels[neighbor_idx] == -1:
            # If previously marked as noise, assign to current cluster
            labels[neighbor_idx] = cluster_id
        elif labels[neighbor_idx] == 0:
            # If unvisited, assign to current cluster
            labels[neighbor_idx] = cluster_id
            new_neighbors = region_query(X, neighbor_idx, eps)
            # If new neighbor is a core point, add its neighbors
            if len(new_neighbors) >= min_samples:
                neighbors.extend(new_neighbors)
        i += 1

def dbscan(X, eps=0.5, min_samples=5):
    labels = np.zeros(len(X), dtype=int)  # 0 means unvisited, -1 means noise
    labels.fill(0)
    cluster_id = 0

    for point_idx in range(len(X)):
        if labels[point_idx] != 0:
            # Skip if already processed
            continue
        neighbors = region_query(X, point_idx, eps)
        if len(neighbors) < min_samples:
            # Not enough neighbors, mark as noise
            labels[point_idx] = -1
        else:
            # Start a new cluster
            cluster_id += 1
            expand_cluster(X, labels, point_idx, neighbors, cluster_id, eps, min_samples)
    return labels