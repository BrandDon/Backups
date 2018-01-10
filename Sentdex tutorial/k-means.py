from random import randrange
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


class ClusteringExample(object):

    def __init__(self):
        self.data = self.make_data()
        self.centroids, self.labels = self.fit()
        self.colors = ["g", "r", "c", "b.", "k.", "o."]
        self.visualize()

    @classmethod
    def make_data(cls, first_midean=10, second_midean=0):
        x = np.array(
            [(randrange(first_midean - 10, first_midean + 10), randrange(first_midean - 8, first_midean + 5)) for _ in
             range(30)] +
            [(randrange(second_midean - 5, second_midean + 3), randrange(second_midean - 5, second_midean + 5)) for _
             in range(30)])
        np.random.shuffle(x)
        return x

    def visualize(self):
        for i in range(len(self.data)):
            plt.scatter(self.data[i][0], self.data[i][1], c=self.colors[self.labels[i]], s=25)
        for centroid in self.centroids:
            plt.scatter(centroid[0], centroid[1], marker='x', s=150, linewidths=35)
        plt.show()

    def fit(self):
        clf = KMeans(n_clusters=3)
        clf.fit(self.data)
        centroids = clf.cluster_centers_
        labels = clf.labels_
        return centroids, labels
