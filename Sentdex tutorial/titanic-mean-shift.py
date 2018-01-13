import numpy as np
import pandas as pd
from matplotlib import style
from sklearn import preprocessing
from sklearn.cluster import MeanShift

style.use('ggplot')

df = pd.read_excel('titanic.xls')
original_df = pd.DataFrame.copy(df)
df.drop(['body', 'name'], 1, inplace=True)
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

clf = MeanShift()
clf.fit(x)

labels = clf.labels_
cluster_centers = clf.cluster_centers_
original_df['cluster_group'] = np.nan
for i in range(len(x)):
    original_df['cluster_group'].iloc[i] = labels[i]

survival_rates = {}
for cluster_num in range(len(cluster_centers)):
    temp_df = original_df[(original_df['cluster_group'] == float(cluster_num))]
    survival_cluster = temp_df[(temp_df['survived'] == 1)]
    survival_rates[cluster_num] = (len(survival_cluster) / float(len(temp_df)) * 100, len(survival_cluster))
print survival_rates
print "Group 0", original_df[(original_df['cluster_group'] == 0)].describe()
print "Group 1", original_df[(original_df['cluster_group'] == 1)].describe()
print "Group 2", original_df[(original_df['cluster_group'] == 2)].describe()
