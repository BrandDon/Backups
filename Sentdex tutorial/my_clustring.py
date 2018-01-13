from random import randrange

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style

style.use('ggplot')


class MyClustering(object):
    def __init__(self, data, k=2, tol=0.0001, max_iter=3000):
        self.data = data
        self.colors = ["g", "r", "c", "b", "k", "o"]
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def visualize(self):
        if not self.centroids or not self.labels:
            print "Please run fit first"
            return
        for label, data_points in self.labels.items():
            for data_point in data_points:
                plt.scatter(data_point[0], data_point[1], c=self.colors[label], s=25)
        for label, centroid in self.centroids.items():
            plt.scatter(centroid[0], centroid[1], marker='x', s=150, linewidths=5, c=self.colors[label])
        plt.show()

    def fit(self):
        self.centroids = {}
        for centroid_num in range(self.k):
            self.centroids[centroid_num] = self.data[centroid_num]

        for iteration in range(self.max_iter):
            self.labels = {}

            for centroid_num in range(self.k):
                self.labels[centroid_num] = []

            # For each of the data points
            for featureset in self.data:
                # find the distance from each centroid
                distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
                # mark the label as the closest centroid (Adds to the cluster)b
                label = distances.index(min(distances))
                self.labels[label].append(featureset)

            prev_centroids = dict(self.centroids)

            for label in self.labels:
                # set each of the centroids to be the average of the cluster
                self.centroids[label] = np.average(self.labels[label], 0)

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                difference = np.abs(np.sum((current_centroid - original_centroid) / original_centroid * 100.0))
                print difference
                if difference > self.tol:  # if the chnage is bigger than the tolerance, if we can tolerate the fact these will be our centroids because the change is so small, it wont break.
                    break
            else:
                return

    def predict(self, featureset):
        # find the distance from each centroid
        distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
        # mark the label as the closest centroid (Adds to the cluster)b
        label = distances.index(min(distances))
        return label


def make_clustering_data(data_points, first_midean=10, second_midean=0):
    x = np.array(
        [(randrange(first_midean - 10, first_midean + 10), randrange(first_midean - 8, first_midean + 5)) for _ in
         range(data_points / 2)] +
        [(randrange(second_midean - 5, second_midean + 3), randrange(second_midean - 5, second_midean + 5)) for _
         in range(data_points / 2)])
    np.random.shuffle(x)
    return x


if __name__ == '__main__':
    clf = MyClustering(make_clustering_data(100))
    clf.fit()

    for data_point in make_clustering_data(10):
        label = clf.predict(data_point)
        plt.scatter(data_point[0], data_point[1], s=100, c=clf.colors[label], edgecolors=clf.colors[3])
    clf.visualize()


