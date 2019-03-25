# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 09:19:51 2019

@author: Bob
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn

from sklearn import model_selection
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

pd.options.display.max_rows = 150

iris = pd.read_csv(
    'C:\\Users\Bob\Documents\Python\IrisID\iris.csv',
    names=[
        'sepal_length',
        'sepal_width',
        'petal_length',
        'petal_width',
        'species'])

print(iris.groupby('species').describe())


"""
From this, we can see that Iris-setosa has by far the smallest petals, both in
length and width.  The maximum value for setosa petal length is 1.9cm, 1.1 cm
shorter than the minimum of versicolor, and a full 2.6cm less than the minimum
for virginica.  It is possible to categorize setosa from the other two by petal
size alone.

Its also pretty clear that the sepal width is least useful in determining
species.  Sepal length has more discrete peaks, but the spreads still overlap
to a large degree.

As a result, setosa will probably be able to be identified with a linear
separation, but the other two will require something more complicated.
"""

# Create a validation dataset
iris_vals = iris.values
X = iris_vals[:,0:4]
Y = iris_vals[:,4]

validation_size = 0.2
seed = 108
scoring = 'accuracy'

X_train, X_valid, Y_train, Y_valid = model_selection.train_test_split(
    X, Y,
    test_size=validation_size,
    random_state=seed)

models = []
models.append((
    'LR',
    LogisticRegression(
        solver='liblinear',
        multi_class='ovr')))

models.append((
    'LDA',
    LinearDiscriminantAnalysis()))

models.append((
    'KNN',
    KNeighborsClassifier()))

models.append((
    'CART',
    DecisionTreeClassifier()))

models.append((
    'NB',
    GaussianNB()))

models.append((
    'SVM',
    SVC(gamma='auto')))

results = []
names = []

kfold = model_selection.KFold(n_splits=10, random_state=seed)
for name, model in models:
    cv_results = model_selection.cross_val_score(
        model,
        X_train,
        Y_train,
        cv=kfold,
        scoring=scoring)
    
    results.append(cv_results)
    names.append(name)
    msg = f"{name}: {cv_results.mean()} {cv_results.std()}"
    print(msg)
    
"""
Clearly, the KNN (K Nearest Neighbors) method is the most effective.  We'll
use that method to categorize the three different types of Iris."""
    
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_valid)
print(accuracy_score(Y_valid, predictions))
print(confusion_matrix(Y_valid, predictions))
print(classification_report(Y_valid, predictions))


