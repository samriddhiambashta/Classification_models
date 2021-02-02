# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 02:53:42 2021

@author: hp
"""


from sklearn import datasets
cancer=datasets.load_breast_cancer()

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(cancer.data,cancer.target,test_size=0.3,random_state=109)
from sklearn import svm
clf=svm.SVC(kernel='linear')
clf.fit(X_train,Y_train)
Y_predict=clf.predict(X_test)
from sklearn import metrics
print(metrics.accuracy_score(Y_test, Y_predict))