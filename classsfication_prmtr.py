import numpy as np
import matplotlib.pyplot as plt
import tarfile
import seaborn as sns
import sys

import os
import pandas as pd
dataset = pd.read_excel('base.xlsx')
del dataset['défaillance']
dataset
dataset.Classe.value_counts()
plt.figure(figsize = (7,7))
corr_ = dataset.corr()
sns.heatmap(corr_, annot = True,linewidths=.5,cmap="YlGnBu");
from sklearn.model_selection import train_test_split #for split the data
from sklearn.metrics import accuracy_score  #for accuracy_score
from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction
from sklearn.metrics import confusion_matrix #for confusion matrix
all_features = dataset.drop("Classe",axis=1)
Targeted_feature = dataset["Classe"]
X_train,X_test,y_train,y_test = train_test_split(all_features,Targeted_feature,test_size=0.2,random_state=None)
X_train.shape,X_test.shape,y_train.shape,y_test.shape
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors = 2)
model.fit(X_train,y_train)
prediction_knn=model.predict(X_test)
#print('--------------The Accuracy of the model----------------------------')
#print('The accuracy of the K Nearst Neighbors Classifier is',round(accuracy_score(prediction_knn,y_test)*100,2))
kfold = KFold(n_splits=2, random_state=None) # k=10, split the data into 10 equal parts
result_knn=cross_val_score(model,all_features,Targeted_feature,cv=2,scoring='accuracy')
#print('The cross validated score for K Nearest Neighbors Classifier is:',round(result_knn.mean()*100,2))
y_pred = cross_val_predict(model,all_features,Targeted_feature,cv=2)
plt.figure(figsize = (9,7))
ax= plt.subplot()
sns.heatmap(confusion_matrix(Targeted_feature,y_pred),annot=True,fmt='5.0f',cmap="summer")
plt.title('Confusion_matrix', y=1, size=15)
ax.xaxis.set_ticklabels(['Classe1','Classe2','Classe3','Classe4','Classe5','Classe6'])
ax.yaxis.set_ticklabels(['Classe1','Classe2','Classe3','Classe4','Classe5','Classe6'])
#print('--------------The Accuracy of the model----------------------------')
#print('The accuracy of the K Nearst Neighbors Classifier is',round(accuracy_score(prediction_knn,y_test)*100,2))
#print('The cross validated score for K Nearest Neighbors Classifier is:',round(result_knn.mean()*100,2))
error = []
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))
plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red' ,linestyle='dashed', marker='o', markerfacecolor='blue' ,markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Eroor')
model.fit(X_train,y_train)
prediction_rm=model.predict(X_test)
def my_model_test(a,b,c):
     z= (model.predict([[a, b, c]]))
     return (z[0])
