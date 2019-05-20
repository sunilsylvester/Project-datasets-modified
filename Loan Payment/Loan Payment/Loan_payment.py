your_local_path = "C:/Users/sylve/Downloads/Machine learning/Dataset/Project datasets modified/"

import pandas as pd
import numpy as np
from sklearn.preprocessing import imupter
from sklearn import tree
from sklearn import metrics
from sklearn import preprocessing
from sklearn import utils
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgridgrid")
% matplotlib inline

df1=pd.read_csv(your_local_path+"Loan payments data.csv")

df1['effective_date']= pd.to_datetime(df1['effective_date'])

df1['due_date'] = pd.to_datetime(df1['due_date'])

df1['paid_off_time'] = pd.to_datetime(df1['paid_off_time'])

from sklearn.preprocessing import LabelEncoder

lb_loan_status = LabelEncoder()
df1["Loan_status_code"] = lb_loan_status.fit_transform(df1["loan_status"])
df1[["loan_status","loan_status_code"]].head(10)

lb = LabelEncoder()
df1["Gender_code"] = lb.fit_transform(df1["Gender"])
df1["education_code"] = lb.fit_transform(df1["education"])

df1[["Gender", "Gender_code"]].head(10)

df1[["education", "education_code"]].head(10) 

df1.head()

df1.info()

df1.describe()

from matplotlib import style
style.use('ggplot')
plt.figure(figsize=(12,4))
sns.boxplot(x="age", data=df1)

df1[df1.age > 48]

kk = []
kt = []
for i in df1["education"].unquie():
	df1_ed = (df1.loc[df1["education"] == i, "Loan_ID"]).count()
	kk.append(df1_ed)
	kt.append(i)
	
print(kk)
print(kt)
kkdf=pd.DataFrame({"level_education":kt,"count":kk})
print(kkdf)

plt.figure(figsize=(8,6))
ax=sns.barplot(x=kkdf["level_education"], y=kkdf["count"])
ax.set_title("Graph of total count VS Education")
ax.set_ylabel('Count')
ax.set_xlabel('Education')

list(df1)

columns = [] incomplete
features_1=df1[list(columns)].values

features_1

my_tree_one = tree.DecisionTreeRegressor(criterion="mse", max_depth=3)
my_tree_one = my_tree_one.fit(features_1, target)






