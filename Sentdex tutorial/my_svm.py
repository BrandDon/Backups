import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('ggplot')


class MySVM(object):
    def __init__(self, data, vis=True):
        self.data = data
        self.colors = {1: 'r', -1: 'b'}
        self.w, self.b, self.max_feature, self.min_feature = [None] * 4
        self.vis = vis
        if vis:
            self.figure = plt.figure()
            self.ax = self.figure.add_subplot(1, 1, 1)
        self.fit()

    def fit(self):
        """
        trains the algorithm to find relevant w and b.
        :return:
        """

        # Opt dict { ||w||: [w,b] }
        opt_dict = {}
        transforms = list()
        for y in (1, -1):
            for x in (1, -1):
                transforms.append((x, y))

        self.max_feature = max(map(lambda x: np.amax(x), self.data.values()))
        self.min_feature = min(map(lambda x: np.amin(x), self.data.values()))

        step_sizes = [self.max_feature * 0.1,
                      self.max_feature * 0.01,
                      self.max_feature * 0.001,
                      self.max_feature * 0.0001]
        # extremely expensive. b takes bigger steps than w, it doesn't need to be as precise.
        b_range_multiple = 2
        #
        b_multiple = 5

        # Corner cut here.
        latest_optimum = self.max_feature * 10
        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])
            # Possible due to convex
            optimized = False
            while not optimized:
                for b in np.arange(-1 * (self.max_feature * b_range_multiple),
                                   self.max_feature * b_range_multiple,
                                   step * b_multiple):
                    for transform in transforms:
                        w_transform = w * transform
                        found_option = True
                        for yi in self.data:
                            for xi in self.data[yi]:
                                if not yi * (np.dot(w_transform, xi) + b) >= 1:
                                    found_option = False
                        if found_option:
                            opt_dict[np.linalg.norm(w_transform)] = [w_transform, b]

                if w[0] < 0:
                    optimized = True
                    print 'optimized a step'
                else:
                    w = w - step

            self.w, self.b = opt_dict[min(opt_dict.keys())]
            latest_optimum = self.w[0] + step * 2

        for yi in self.data:
            for xi in self.data[yi]:
                print "{0}: {1}".format(xi, yi * (np.dot(self.w, xi) + self.b))

    def predict(self, x):
        # the sign of (w*x + b) marks the correct classification of point x
        result = np.sign(np.dot(np.array(x), self.w) + self.b)
        if result != 0 and self.vis:
            self.ax.scatter(x[0], x[1], s=200, marker='*', c=self.colors[result])
        return result

    @classmethod
    def _vis_hyperplane(cls, x, w, b, v):
        return (-w[0] * x - b + v) / w[1]

    def visualize(self):
        [[self.ax.scatter(x[0], x[1], s=100, color=self.colors[i]) for x in self.data[i]] for i in self.data.keys()]
        data_range = (self.min_feature * 0.9, self.max_feature * 1.1)
        hyp_x_min = data_range[0]
        hyp_x_max = data_range[1]

        # psv
        psv1 = self._vis_hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = self._vis_hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min, hyp_x_max], [psv1, psv2])

        # nsv
        nsv1 = self._vis_hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = self._vis_hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2])

        # db
        db1 = self._vis_hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = self._vis_hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min, hyp_x_max], [db1, db2])
        plt.show()

    # hyperplane = w*x+b
    # v = w*x +b
    # psv = 1
    # nsv = -1
    # decision boundary = 0


data_points = {-1: np.array([[3, 9], [2, 10], [6, 8]]),
               1: np.array([[2, 0], [-1, -1], [4, 3]])}
svm = MySVM(data_points)
predict = [[7, 10], [1, 3], [4, 10], [-3, 10]]
for p in predict:
    svm.predict(p)
svm.visualize()
