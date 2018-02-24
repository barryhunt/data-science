# -*- coding: utf-8 -*-
# /* Created by PC on 2018/2/23.*/

from sklearn import preprocessing #标准化数据模块
import numpy as np
from sklearn.model_selection import train_test_split # 将data分割成train与test的模块
from sklearn.datasets.samples_generator import make_classification #生成适合做classification的数据的模块
from sklearn.svm import SVC
import matplotlib.pyplot as plt

a = np.array([[10, 2.7, 3.6],
                     [-100, 5, -2],
                     [120, 20, 40]], dtype=np.float64)
print(a)
print(preprocessing.scale(a)) #将normalized后的a打印出

#生成具有2种属性的300笔数据
X, y = make_classification(n_samples=300, n_features=2 , n_redundant=0, n_informative=2,
                           random_state=22, n_clusters_per_class=1, scale=100)
#可视化标准化后的数据，并输出评分
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()
X = preprocessing.scale(X)    # 正则化步骤
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
clf = SVC()
clf.fit(X_train, y_train)
print (clf.score(X_test, y_test))