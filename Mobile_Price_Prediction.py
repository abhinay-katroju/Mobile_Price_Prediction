import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

df = pd.read_csv('/content/mobile price predictions.csv')
df.head()

df.shape

df.describe()

df.info()

plt.figure(figsize = (10,10))
sns.heatmap(df.corr())
plt.show

plt.figure(figsize = (11,6))
sns.barplot(x = 'price_range', y = 'battery_power', data = df)
plt.show()

plt.figure(figsize = (11,6))
plt.subplot(1,2,1)
sns.barplot(x = 'price_range', y = 'px_height', data = df, palette = 'Reds')
plt.subplot(1,2,2)
sns.barplot(x = 'price_range', y = 'px_width', data = df, palette = 'Blues')
plt.show()

plt.figure(figsize = (10,5))
sns.barplot(x = 'price_range', y = 'ram', data = df)
plt.show()

plt.figure(figsize = (11,6))
sns.countplot(x='three_g', hue='price_range', data=df, palette='pink') # Use x and hue parameters with data
plt.show()

plt.figure(figsize = (11,6))
sns.countplot(x='four_g', hue='price_range', data=df, palette='ocean')
plt.show()

plt.figure(figsize = (11,6))
sns.lineplot(x='price_range', y='int_memory', data=df , hue = 'dual_sim')
plt.show()

x = df.drop(['price_range'] , axis = 1)
y = df['price_range']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 10)
knn.fit(x_train, y_train)

KNeighborsClassifier(n_neighbors=10)

knn.score(x_train, y_train)

predictions = knn.predict(x_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)

input_data= (1821,0,1.7,0,4,1,10,0.8,139,8,10,381,1018,3220,13,8,18,1,0,1)
input_data_as_numpy_array=np.asarray(input_data)

input_data_reshape=input_data_as_numpy_array.reshape(1,-1)

prediction=knn.predict(input_data_reshape)
print(prediction)
