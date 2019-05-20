your_local_path = "C:/Users/sylve/Downloads/Machine learning/Dataset/Project datasets modified/"

import pandas as pd
import numpy as np
from sklearn.preprocessing import imupter
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgridgrid")
% matplotlib inline

df1= pd.read_csv(your_local_path+"Supermarket Purchase.csv")

df1.info()

df1.describe()

print("total values")
print(df1["cust_id"].count())
print("\n Max Values for all the attribute")
print(df1.loc[df1["No_of_Items"]].max())
print("\n Row of max values for column no_of_items")
print(df1.loc[df1["No_of_Items"].idxmax()])

df1.head(10)

df1_train,df1_test = train_test_split(df1,test_size=0.33, random_state=42)

print(df1_train.head())
print(df1_test.head())

print("****** Train set *******")
print(df1_train.describe())
print("\n")
print("****** Test set *******")
print(df1_test.describe())

from sklearn.preprocessing import MinMaxScaler
scld = MinMaxScaler(feature_range=(0,1))
arr_scld = scld.fit_transform(df1_train)
df1_train_scld = pd.DataFrame(arr_scld, columns = df1_train.columns)
df1_train_scld.head()

num_clusters = range(2,10)
error = []
for i in num_clusters:
	cluster = KMeans(i)
	cluster.fit(df1_train_scld)
	error.append(cluster.inertia_/100)
	
df = pd.DataFrame({"cluster_number":num_clusters, "Error_term":error})
print(df)

df1_train_scld["cluster_label"] = cluster.label_
df1_train_scld.head(50)

plt.figure(figsize=(10,8))
plt.plot(df.cluster_number,df.Error_term, marker = "D" color="red")
plt.show()

clusters1 = KMeans(4)
clusters1.fit(df1_train_scld)
clusters1.labels_

clusters1.inertia_/100























