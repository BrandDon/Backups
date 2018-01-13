from random import randrange

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
from sklearn.datasets import make_blobs


style.use('ggplot')


class MyMeanShift(object):
    def __init__(self, data, radius=None, radius_norm_step=300):
        self.data = data
        self.colors = ["g", "r", "c", "b"] * 10
        self.radius = radius
        self.radius_norm_step = radius_norm_step

    def visualize(self):
        if not self.centroids:
            print "Please run fit first"
            return
        for data_point in self.data:
            plt.scatter(data_point[0], data_point[1], s=25)
        for label, centroid in self.centroids.items():
            print self.colors[label]
            plt.scatter(centroid[0], centroid[1], marker='x', s=150, linewidths=5, c=self.colors[label])
        plt.show()

    def fit(self):
        if not self.radius:
            all_data_centroid = np.average(self.data, axis=0)
            all_data_norm = np.linalg.norm(all_data_centroid)
            self.radius = all_data_norm / self.radius_norm_step

        centroids = {centroid_id: data_point for centroid_id, data_point in enumerate(self.data)}
        weights = [x**2 / 2 for x in range(self.radius_norm_step)][::-1]

        while True:
            new_centroids = []
            for centroid_id, centroid in centroids.items():
                in_bandwidth = []

                for featureset in self.data:
                    distance = np.linalg.norm(featureset - centroid)
                    weight_index = int(distance / self.radius)
                    if weight_index > len(weights)-1:
                        weight_index = len(weights)-1
                    to_add = (weights[weight_index])*[featureset]
                    in_bandwidth += to_add


                new_centroid = np.average(in_bandwidth, axis=0)
                new_centroids.append(tuple(new_centroid))

            uniques = sorted(list(set(new_centroids)))
            to_pop = set()
            for unique in uniques:
                for other_unique in uniques:
                    if unique in to_pop:
                        break
                    if unique == other_unique:
                        continue
                    if np.linalg.norm(np.array(unique)-np.array(other_unique)) <= self.radius:
                        to_pop.add(other_unique)
            for c in to_pop:
                uniques.remove(c)
            prev_centroids = dict(centroids)
            centroids = {}
            for i in range(len(uniques)):
                centroids[i] = np.array(uniques[i])
            for cid in centroids:
                if not np.array_equal(centroids[cid], prev_centroids[cid]):
                    break
            else:
                break

            self.centroids = centroids

    def predict(self, featureset):
        pass


def make_clustering_data(data_points, first_midean=10, second_midean=0):
    x = np.array(
        [(randrange(first_midean - 10, first_midean + 10), randrange(first_midean - 8, first_midean + 5)) for _ in
         range(data_points / 2)] +
        [(randrange(second_midean - 5, second_midean + 3), randrange(second_midean - 5, second_midean + 5)) for _
         in range(data_points / 2)])
    np.random.shuffle(x)
    return x


if __name__ == '__main__':
    data = make_blobs(24, n_features=2)
    clf = MyMeanShift(data)
    clf.fit()
    clf.visualize()
