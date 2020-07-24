# -*- coding: utf-8 -*- 
"""
Author: 18120900
Created: 2020/6/22 9:11
Software: PyCharm
Desc: 
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
X_train, X_test, Y_train, Y_test = train_test_split(df[iris.feature_names], df['target'], random_state=0)
clf = DecisionTreeClassifier(max_depth=2, random_state=0)
clf.fit(X_train, Y_train)
fn = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
cn = ['setosa', 'versicolor', 'virginica']
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))
tree.plot_tree(clf, feature_names=fn, class_names=cn, filled=True)
fig.savefig('imagename.png')
plt.show()
