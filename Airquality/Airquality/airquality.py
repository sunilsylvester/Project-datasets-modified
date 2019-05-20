your_local_path = "C:/Users/sylve/Downloads/Machine learning/Dataset/Project datasets modified/"

import pandas as pd
from sklearn.preprocessing import imupter
from sklearn import tree
from sklearn import metrics
from sklearn import preprocessing
from sklearn import utils
import numpy as np
import matplotlib.pyplot as plt
% matplotlib inline

train_df = pd.read_csv(your_local_path+"AirQualityUCI.csv")

train_df.head()

train_df = train_df.replace(-200,np.nan)

train_df = train_df.dropna(thresh = 3)

train_df.shape

train_df1 = train_df.drop(['NMHC(GT)'],axis=1)

train_df1.info()

train_df1.describe()

col1 = ["RH"]
target=train_df1[list(col1)].values
target

target.shape

imp1= Imputer(missing_values = 'NaN', strategy='mean', axis=0)
y=imp1.fit_transform(target)

y.shape

columns=["CO(GT)","PT0.S1(CO)"]   incomplete
features = train_df1[list(columns)].values
features

imp= Imputer(missing_values = 'NaN', strategy='mean', axis=0)
X=imp.fit_transform(features)
X

X.shape

my_tree_one = tree.DecisionTreeRegressor(criterion="mse", max_depth=3)
my_tree_one = my_tree_one.fit(X,y)

my_tree_one

print(my_tree_one.feature_inportances_)
print(my_tree_one.score(X,y))
list(zip(columns,my_tree_one.feature_inportances_))

test_df = pd.read_csv(your_local_path+"AirQualityUCI.csv")

test_df = test_df.replace(-200,np.nan)
test_df = test_df.drop(['NMHC(GT)'],axis=1)

col1 = ["RH"]
target_test = test_df[list(col_test)].values
target_test

imp_test= Imputer(missing_values = 'NaN', strategy='mean', axis=0)
y_test=imp_test.fit_transform(target_test)
y_test

columns_test=["CO(GT)","PT0.S1(CO)"]   incomplete
features_test = test_df[list(columns_test)].values
features_test

imp= Imputer(missing_values = 'NaN', strategy='mean', axis=0)
X_test=imp.fit_transform(features)
X_test

pred = my_tree_one.predict(X_test)
df_mse = metrics.mean_squared_error(y,pred)
df_mse

pred.shape





