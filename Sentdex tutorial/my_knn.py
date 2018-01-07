import random
import warnings
from collections import Counter
import pandas as pd
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
        self.colors = {2: 'k', 4:'r'}

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
        biggest_group = Counter(votes).most_common(1)[0][0]
        confidence = float(votes.count(biggest_group)) / self.k * 100
        if biggest_group == 0:
            biggest_group = 2
        else:
            biggest_group = 4
        return biggest_group, confidence

    def show_graph(self):
        [[plt.scatter(point[0], point[1], s=100, color=self.colors[i]) for point in table] for i, table in
         enumerate(self.data)]
        plt.show()

    def apply_new_point(self, point=(3, 5)):
        group_to_add = self.k_nearest_neighbor(point)
        plt.scatter(point[0], point[1], s=100, color=self.colors[group_to_add])
        self.show_graph()


df = pd.read_csv('breast-cancer-wisconsin.csv')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

full_data = df.astype(float).values.tolist()

random.shuffle(full_data)
test_size = 0.2
train_set = {2: [], 4: []}
test_set = {2: [], 4: []}
train_data = full_data[:-int(test_size * len(full_data))]
test_data = full_data[-int(test_size * len(full_data)):]

for data_point in train_data:
    train_set[data_point[-1]].append(data_point[:-1])

for data_point in test_data:
    test_set[data_point[-1]].append(data_point[:-1])

correct = 0
total = 0
confidences = []
k = MyKNearestNeighbours(data=[
    train_set[2],
    train_set[4]], k=5)
for group in test_set:
    for data_point in test_set[group]:
        vote, conf = k.k_nearest_neighbor(data_point)
        if vote == group:
            correct += 1
        else:
            confidences.append(conf)
        total += 1

conf_avg = sum(confidences) / len(confidences)
accuracy = float(correct) / float(total)
print "Accuracy: {0}%".format(accuracy * 100)
print conf_avg