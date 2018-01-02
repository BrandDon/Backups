import warnings
from collections import Counter

from statistics import mean
import numpy
import matplotlib.pyplot as plt
from matplotlib import style
from math import sqrt

style.use('fivethirtyeight')


class MyKNearestNeighbours(object):
    def __init__(self, data, k=3):
        self.data = data
        self.k = k
        self.colors = ['k', 'r']

        if len(data) >= self.k:
            warnings.warn("K is smaller than the total voting groups")

        self.number_of_features = len(data[0][0])
        assert (all([len(point) == self.number_of_features for point in group] for group in
                    self.data))  # Check every point has the same number of features

    @classmethod
    def _calc_euclidean_d(cls, point1, point2):
        euclidean_distance = numpy.linalg.norm(numpy.array(point1) - numpy.array(point2))
        return euclidean_distance

    def k_nearest_neighbor(self, point_to_check):
        distances = []
        for group_num, group in enumerate(self.data):
            for point in group:
                distances.append([self._calc_euclidean_d(point, point_to_check), group_num])

        votes = [distance_and_group[1] for distance_and_group in sorted(distances)[:self.k]]
        print votes
        biggest_group = Counter(votes).most_common(1)[0][0]

        return biggest_group

    def show_graph(self):
        [[plt.scatter(point[0], point[1], s=100, color=self.colors[i]) for point in table] for i, table in
         enumerate(self.data)]
        plt.show()

    def apply_new_point(self, point=(3, 5)):
        group_to_add = self.k_nearest_neighbor(point)
        plt.scatter(point[0], point[1], s=100, color=self.colors[group_to_add])
        self.show_graph()


k = MyKNearestNeighbours(data=[
    [[1, 2], [3, 4]],
    [[5, 6], [6, 7]]])
