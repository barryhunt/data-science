# -*- coding: utf-8 -*-
# /* Created by PC on 2018/2/23.*/
'''
使用Learning curve 观察过拟合
sklearn.learning_curve 中的 learning curve 可以很直观的看出我们的 model 学习的进度, 对比发现有没有 overfitting 的问题. 然后我们可以对我们的 model 进行调整, 克服 overfitting 的问题
'''

from sklearn.learning_curve import  learning_curve  #学习曲线模块
from sklearn.datasets import load_digits   #digits数据集
'''
加载digits数据集，其包含的是手写体的数字，从0到9。数据集总共有1797个样本，每个样本由64个特征组成， 分别为其手写体对应的8×8像素表示，每个特征取值0~16
'''
from sklearn.svm import SVC     #Support Vector Classifier
import matplotlib.pyplot as plt     #可视化模块
import numpy as np

digits = load_digits()
X = digits.data
y = digits.target
train_sizes, train_loss, test_loss= learning_curve(
        SVC(gamma=0.01), X, y, cv=10, scoring='mean_squared_error',
        train_sizes=[0.1, 0.25, 0.5, 0.75, 1])
'''
观察样本由小到大的学习曲线变化, 采用K折交叉验证 cv=10, 选择平均方差检视模型效能 scoring='mean_squared_error', 样本由小到大分成5轮检验学习曲线(10%, 25%, 50%, 75%, 100%)
'''
train_loss_mean = -np.mean(train_loss, axis=1)
test_loss_mean = -np.mean(test_loss, axis=1)


##可视化图形
############
plt.plot(train_sizes, train_loss_mean, 'o-', color="r",
             label="Training")
plt.plot(train_sizes, test_loss_mean, 'o-', color="g",
             label="Cross-validation")

plt.xlabel("Training examples")
plt.ylabel("Loss")
plt.legend(loc="best")
plt.show()