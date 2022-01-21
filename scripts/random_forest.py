'''

This script implements a random forest regressor and uses it to predict on data

'''
#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap


print("Successfully imported libraries")

#lets import the data

try:
    dataset=pd.read_csv("../data/mobile_ads.csv")
    print("Fetched dataset")
except FileNotFoundError:
    print("File not found \n Exiting program")
    sys.exit(1)
    
except Exception as e:
    print("Unexpected error:", sys.exc_info()[0])
    sys.exit(1)
    
#lets view the data
dataset.head()
#lets split the data into training and testing data

X=dataset.drop(labels=['User ID','Gender','Purchased'],axis=1)
X=X.values
y=dataset['Purchased'].values

print(X)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#we scale our features to have a standard scale between 0 and 1
ss=StandardScaler()

#fit the scaler to the training data and only transform the test data
X_train=ss.fit_transform(X_train)
X_test=ss.transform(X_test)


#lets build our random forest Classifier 

rf=RandomForestClassifier(n_estimators=150,criterion='entropy',random_state=0)

print("Training the model")
rf.fit(X_train,y_train)

print("Done ! ")

#lets make predictions 

print("Making predictions")
y_pred=rf.predict(X_test)

print("Done ! ")

#we draw up a confusion matrix to display our model scores
cm=confusion_matrix(y_test,y_pred)
print(cm)

X_Set,Y_Set=X_train,y_train

X1,X2=np.meshgrid(np.arange(start=X_Set[:,0].min()-1,stop=X_Set[:,0].max()+1,step=0.01),
                   np.arange(start=X_Set[:,1].min()-1,stop=X_Set[:,1].max()+1,step=0.01))
plt.contour(X1,X2,rf.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),alpha =0.75, cmap=ListedColormap(('red','green'))) 
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())

for i ,j in enumerate(np.unique(Y_Set)):
    plt.scatter(X_Set[Y_Set ==j, 0], X_Set[Y_Set == j ,1],
                color=ListedColormap(('red','green'))(i), label =j)

plt.title("RandomForestClassifier on training data")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()


X_Set,Y_Set=X_test,y_test

X1,X2=np.meshgrid(np.arange(start=X_Set[:,0].min()-1,stop=X_Set[:,0].max()+1,step=0.01),
                   np.arange(start=X_Set[:,1].min()-1,stop=X_Set[:,1].max()+1,step=0.01))
plt.contour(X1,X2,rf.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),alpha =0.75, cmap=ListedColormap(('red','green'))) 
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())

for i ,j in enumerate(np.unique(Y_Set)):
    plt.scatter(X_Set[Y_Set ==j, 0], X_Set[Y_Set == j ,1],
                color=ListedColormap(('red','green'))(i), label =j)

plt.title("RandomForestClassifier on test data")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()
# %%
