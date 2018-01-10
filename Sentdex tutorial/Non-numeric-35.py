import numpy as np
import pandas as pd
from matplotlib import style
from sklearn import preprocessing
from sklearn.cluster import KMeans
from my_clustring import MyClustering, make_clustering_data
style.use('ggplot')

df = pd.read_excel('titanic.xls')
df.drop(['body', 'pclass', 'name', 'boat'], 1, inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)


def handle_non_numerical_data(dataframe):
    columns = dataframe.columns.values
    for column in columns:
        text_digit_vals = {}

        def convert_to_int(val):
            return text_digit_vals[val]

        if dataframe[column].dtype != np.int64 and dataframe[column].dtype != np.float64:
            column_contents = dataframe[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for element in unique_elements:
                if element not in text_digit_vals:
                    text_digit_vals[element] = x
                    x += 1
            dataframe[column] = list(map(convert_to_int, dataframe[column]))

    return dataframe


df = handle_non_numerical_data(df)

x = np.array(df.drop(['survived'], 1).astype(float))
x = preprocessing.scale(x)
y = np.array(df['survived'])

clf = MyClustering(x)
clf.fit()

correct = 0
for i in range(len(x)):
    predict = np.array(x[i].astype(float))
    predict = predict.reshape(-1, len(predict))
    prediction = clf.predict(predict)
    if prediction == y[i]:
        correct += 1
accuracy = float(correct) / len(x) * 100
if accuracy < 50:
    accuracy = 100 - accuracy
print accuracy
