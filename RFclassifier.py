# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 19:37:01 2021

@author: hp
"""


from sklearn import datasets
iris=datasets.load_iris()

import pandas as pd
import numpy as np
data=pd.DataFrame({'sepallength': iris.data[:,0],'sepalwidth': iris.data[:,1],'petallength': iris.data[:,2],'petalwidth': iris.data[:,3], 'species': iris.target})
#print(data.head())
X=data[['sepallength', 'sepalwidth', 'petallength', 'petalwidth']]
Y=data['species']

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train,Y_train)
Y_predict=clf.predict(X_test)

from sklearn import metrics
print('Accuracy: ', metrics.accuracy_score(Y_test, Y_predict))
print(clf.predict([[3,5,4,2]]))
sl=float(input('Enter the sepal length: '))
sw=float(input('Enter the sepal width: '))
pl=float(input('Enter the petal length: '))
pw=float(input('Enter the petal width: '))
user_data=np.array([sl,sw,pl,pw])
sp=clf.predict(user_data.reshape(1,-4))
print(sp)




