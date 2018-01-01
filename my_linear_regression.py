from statistics import mean
import numpy
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')


class MyLinearRegression():
    def __init__(self, var):
        self.all_x, self.all_y = self.create_dataset(100, var,correlatiaon=True)
        self.m = self.best_fit_slope()
        self.b = self.best_fit_interception()
        self.regression_line = self.make_line()
        self.r_sqr = self.calc_accuracy()

    def create_dataset(self, dp_amount, variance, step=2, correlatiaon=False):
        val = 1
        all_y = []
        for dp in range(dp_amount):
            y = val + random.randrange(-variance, variance)
            all_y.append(y)
            if correlatiaon:
                val += step
        return numpy.array(range(dp_amount), dtype=numpy.float64), numpy.array(all_y, dtype=numpy.float64)

    def best_fit_slope(self):
        m = mean(self.all_x) * mean(self.all_y)
        m = m - mean(self.all_y * self.all_x)
        m = m / (mean(self.all_x) ** 2 - mean([x ** 2 for x in self.all_x]))
        return m

    def best_fit_interception(self):
        b = mean(self.all_y) - self.m * mean(self.all_x)
        return b

    def show_graph(self):
        plt.scatter(self.all_x, self.all_y)
        plt.plot(self.all_x, self.regression_line)
        plt.show()

    def make_line(self):
        regression_line = [(self.m * x) + self.b for x in self.all_x]
        return numpy.array(regression_line)

    def calc_squared_error(self, line_to_check):
        return sum((line_to_check - self.all_y) ** 2)  # vector substraction is possible because all_y is a numpy array

    def calc_accuracy(self):
        mean_y_line = numpy.array([mean(self.all_y)] * len(self.all_y))
        r_squared = (self.calc_squared_error(self.regression_line))
        r_squared = r_squared / self.calc_squared_error(mean_y_line)
        r_squared = 1 - r_squared
        return r_squared


lin = MyLinearRegression(5)
print lin.r_sqr
lin.show_graph()

