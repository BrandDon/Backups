import pandas as pd
import quandl
import math
import numpy
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import datetime
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')


df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df[
    'Adj. Low'] * 100.0  # Gets the precentage of difference between the high and low
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df[
    'Adj. Open'] * 100.0  # Gets the precentage of difference between the closing and opening


df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]  # the features
forecast_column = 'Adj. Close'  # what we are trying to predict

df.fillna(-99999, inplace=True)  # Replace the bad data (Extreme data will  be ignored?)

forecast_out = 200  # Days in advance to predict.

df['label'] = df[forecast_column].shift(-forecast_out)  # label all but the last 'forecast out'
X = numpy.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)  # drop all the incompletes (only those unlabeled)
Y = numpy.array(df['label'])

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.2)

clf = LinearRegression(n_jobs=10)
clf.fit(X_train, Y_train)

#Saving and loading the trained classifier:
with open('linearregression.pickle', 'wb') as f:
    pickle.dump(clf, f)
pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)


accuracy = clf.score(X_test, Y_test)

forecast_set = clf.predict(X_lately)

print(forecast_set, accuracy, forecast_out)

df['Forecast'] = numpy.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day_seconds = 86400
next_unix = last_unix + one_day_seconds
for forecast in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day_seconds
    df.loc[next_date] = [numpy.nan for _ in range(len(df.columns) - 1)] + [forecast]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
