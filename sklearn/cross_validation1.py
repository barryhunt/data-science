# -*- coding: utf-8 -*-
# /* Created by PC on 2018/2/23.*/

'''
Cross Validation (交叉验证)对于我们选择正确的 Model 和 Model 的参数是非常有帮助
以下为使用交叉验证，选择knn的最优K值
'''
########  Model基础评价
#####################
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
X = iris.data
y = iris.target

# 分割数据集#
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)
#建立knn模型，这里指定为5
knn = KNeighborsClassifier(n_neighbors=5)
#训练模型
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(knn.score(X_test, y_test))  #基础评价为0.973684210526

########Model 交叉验证法
#######################
#K折交叉验证模块
from sklearn.cross_validation import cross_val_score

knn = KNeighborsClassifier(n_neighbors=5)
#使用K折交叉验证模块
scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
#将5次的预测准确率打印出
print(scores)
print(scores.mean())
# 0.973333333333

#####如何使用交叉验证来选择模型和配置
###############################
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt

#建立测试参数集
k_range = range(1, 31)
k_scores = []
#藉由迭代的方式来计算不同参数对模型的影响，并返回交叉验证后的平均准确率
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
#loss = -cross_val_score(knn, X, y, cv=10, scoring='mean_squared_error')
#  #对于回归问题，使用平均方差
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy') # 对于分类问题，使用准确率
    k_scores.append(scores.mean())

plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()
