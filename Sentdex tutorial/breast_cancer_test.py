import numpy as np
from sklearn import preprocessing, model_selection, neighbors, svm
import pandas

df = pandas.read_csv('breast-cancer-wisconsin.csv')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

x = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2)

clf = svm.SVC()
clf.fit(x_train, y_train)

accuracy = clf.score(x_test, y_test)

print accuracy

example_measures = np.array([[4, 2, 1, 1, 2, 1, 3, 1, 2], [1, 3, 2, 1, 2, 1, 3, 3, 10]])
example_measures = example_measures.reshape(len(example_measures), -1)

pred = clf.predict(example_measures)
print pred
