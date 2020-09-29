#-*- codeing=utf-8 -*-
#@time: 2020/9/29 15:45
#@Author: Shang-gang Lee

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import pandas as pd

data,label=load_iris(return_X_y=True)
RFC=RandomForestClassifier(n_estimators=9,criterion='gini')
param={'max_features':[i for i in range(1,5)]}
clf=GridSearchCV(estimator=RFC,param_grid=param,cv=5,scoring='accuracy',n_jobs=-1,return_train_score=True)
clf.fit(data,label)
print("best_params:",clf.best_params_)
print('best_score:',clf.best_score_)
print(clf.cv_results_)
results=pd.DataFrame(clf.cv_results_)
results.to_csv('max_features-result.xls',index=False)

#visdom
max_features=pd.read_csv('max_features-result.xls')

mean_test_score=max_features['mean_test_score']
std_test_score=max_features['std_test_score']
mean_train_score=max_features['mean_train_score']
std_train_score=max_features['std_train_score']

plt.figure(figsize=(12,8))
plt.plot(mean_test_score)
plt.plot(mean_train_score)
plt.legend(['mean_test_score','mean_train_score'])
plt.title('fine-tuning-max_features')
plt.xlabel('max_features')
plt.ylabel('std-score')
plt.xticks(max_features.param_max_features)
plt.show()

plt.figure(figsize=(12,8))
plt.plot(std_test_score)
plt.plot(std_train_score)
plt.legend(['std_test_score','std_train_score'])
plt.title('fine-tuning-max_features')
plt.xlabel('max_features')
plt.ylabel('mean-score')
plt.xticks(max_features.param_max_features)
plt.show()
