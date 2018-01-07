import pickle

# import matplotlib.pyplot as plt
import numpy
import pandas as pd
from matplotlib import style
from sklearn import model_selection, ensemble

style.use('ggplot')

df = pd.read_csv('decks.csv')
# df['Mean'] = (df['Cost-0'] * 0 + df['Cost-1'] * 1 + df['Cost-2'] * 2 + df['Cost-3'] * 3 + df['Cost-4'] * 4 + df[
#     'Cost-5'] * 5 + df['Cost-6'] * 6 + df['Cost-7+'] * 8.3) / 8
X = numpy.array(df.drop(['Type'], 1))
# X = preprocessing.scale(X)

Y = numpy.array(df['Type'])

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2)

clf = ensemble.RandomForestClassifier()
clf.fit(X_train, Y_train)

# Saving and loading the trained classifier:
with open('trained_deck_clf.pickle', 'wb') as f:
    pickle.dump(clf, f)
# pickle_in = open('linearregression.pickle', 'rb')
# clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, Y_test)

print 'Accuracy: {0}'.format(accuracy)

# legend = {
#     'Midrange': (0, 'b'),
#     'Mid-Range': (0, 'b'),
#     'Aggro': (1, 'r'),
#     'Control': (2, 'y')
# }
# i=0
# for x, y in zip(df['Mean'], df['Type']):
#     i += 1
#     plt.scatter(x, legend[y][0], s=100, color=legend[y][1])
#     if i > 1000:
#         break
# plt.show()
