import random
from math import sqrt


def dot_product(v1, v2):
    """Get the dot product of the two vectors.
    if A = [a1, a2, a3] && B = [b1, b2, b3]; then
    dot_product(A, B) == (a1 * b1) + (a2 * b2) + (a3 * b3)
    true
    Input vectors must be the same length.
    """
    return sum(a * b for a, b in zip(v1, v2))


def magnitude(vector):
    """Returns the numerical length / magnitude of the vector."""
    return sqrt(dot_product(vector, vector))


def similarity(v1, v2):
    """Ratio of the dot product & the product of the magnitudes of vectors."""
    return dot_product(v1, v2) / (magnitude(v1) * magnitude(v2) + .00000000001)


class KMeans(object):
    """K-Means clustering. Uses cosine similarity as the distance function."""

    def __init__(self, k, vectors):
        assert len(vectors) >= k
        self.centers = random.sample(vectors, k)
        self.clusters = [[] for c in self.centers]
        self.vectors = vectors

    def update_clusters(self):
        """Determine which cluster center each `self.vector` is closest to."""
        def closest_center_index(vector):
            """Get the index of the closest cluster center to `self.vector`."""
            similarity_to_vector = lambda center: similarity(center,vector)
            center = max(self.centers, key=similarity_to_vector)
            return self.centers.index(center)

        self.clusters = [[] for c in self.centers]
        for vector in self.vectors:
             index = closest_center_index(vector)
             self.clusters[index].append(vector)

    def update_centers(self):
        """Move `self.centers` to the centers of `self.clusters`.
        Return True if centers moved, else False.
        """
        new_centers = []
        for cluster in self.clusters:
            center = [average(ci) for ci in zip(*cluster)]
            new_centers.append(center)

        if new_centers == self.centers:
            return False

        self.centers = new_centers
        return True

    def main_loop(self):
        """Perform k-means clustering."""
        self.update_clusters()
        while self.update_centers():
            self.update_clusters()


def average(sequence):
    return sum(sequence) / len(sequence)