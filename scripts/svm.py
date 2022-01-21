''''
This script implements a scale vector machine algorithm for classification.

'''
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

import sklearn.preprocessing as preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import seaborn as sns
import sys
import os

print("Libraries successfully loaded.")

#we load our dataset

try:
    print("Reading data")
    data=pd.read_csv("../data/applicant.csv",
                     names=["Age","Workclass","Fnlwgt","Education","Education-Num","Marital Status","Occupation","Relationship","Race","Gender","Capital_Gain","Capital_Loss","Hours_per_week","Country","Target"],
                     )
    print(data.head())

except FileNotFoundError:
    print("File not found")
    sys.exit(1)
except Exception as e:
    print("Error occurred")
    sys.exit(1)
# %%
#we plot a graph for our variables
fig=plt.figure(figsize=(20,20))
cols=3

rows=math.ceil(float(data.shape[1]/cols))

# %%
for i, column in enumerate(['Age','WorkClass','Education','Occupation','Race','Gender']):
    ax=fig.add_subplot(rows,cols,i+1)
    ax.set_title(column)
    if data.dtypes[column] == np.object:
        data[column].value_counts().plot(kind="bar",axes=ax)
    else:
        data[column].hist(axes=ax)
        plt.xticks(rotation='vertical')

plt.subplots_adjust(hspace=0.7, wspace=0.2)
plt.show()        
# %%
#we label encode our categorical columns
le=preprocessing.LabelEncoder()
data['Occupation']=le.fit_transform(data['Occupation'].astype(str))
data.head()

 
# %%
data['Marital Status']=le.fit_transform(data['Marital Status'].astype(str))
data.head()
# %%
data['Target']=le.fit_transform(data['Target'].astype(str))
data.head()
# %%
#generate a graph from the data

data.groupby('Education-Num').Target.mean().plot(kind="bar")
plt.show()

# %%
X=data[['Education-Num','Age','Capital_Gain','Capital_Loss','Hours_per_week','Occupation']].values
y=data['Target'].values

#we split our data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print(X_train[:10])
# %%
svc=SVC()
svc.fit(X_train, y_train)
score=svc.score(X_train, y_train)
print(score)
# %%

sns.heatmap(data.corr(),annot=True)
plt.show()
# %%
#we recalibrate our variables to see if they make any difference in our model
X=data[['Education-Num','Age','Capital_Gain','Capital_Loss','Hours_per_week','Marital Status']].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print(X_train[:10])
# %%
svc=SVC()
svc.fit(X_train, y_train)
score=svc.score(X_train, y_train)
print(score)

# %%
data['Gender']=le.fit_transform(data['Gender'].astype(str))
X=data[['Education-Num','Occupation','Age','Gender']].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print(X_train[:10])
# %%
svc=SVC()
svc.fit(X_train, y_train)
score=svc.score(X_train, y_train)
print(score)
# %%
classifier =SVC(kernel='rbf', C=1.0)
classifier.fit(X_train,y_train)
score=classifier.score(X_test,y_test)
print(score)
# %%
classifier =SVC(kernel='linear', C=10.0)
classifier.fit(X_train,y_train)
score=classifier.score(X_test,y_test)
print(score)

# %%
