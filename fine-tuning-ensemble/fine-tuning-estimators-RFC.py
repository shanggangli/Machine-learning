#-*- codeing=utf-8 -*-
#@time: 2020/9/28 22:52
#@Author: Shang-gang Lee

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import pandas as pd

data,label=load_iris(return_X_y=True)
RFC=RandomForestClassifier()
param={'n_estimators':[i for i in range(1,100)]}
clf=GridSearchCV(estimator=RFC,param_grid=param,cv=5,scoring='accuracy',n_jobs=-1,return_train_score=True)
clf.fit(data,label)
print("best_params:",clf.best_params_)
print('best_score:',clf.best_score_)
print(clf.cv_results_)
results=pd.DataFrame(clf.cv_results_)
results.to_csv('n_estimators-result.xls',index=False)

#visdom
n_estimators_result=pd.read_csv('n_estimators-result.xls')

mean_test_score=n_estimators_result['mean_test_score']
std_test_score=n_estimators_result['std_test_score']
mean_train_score=n_estimators_result['mean_train_score']
std_train_score=n_estimators_result['std_train_score']

plt.figure(figsize=(12,8))
plt.plot(mean_test_score)
plt.plot(mean_train_score)
plt.legend(['mean_test_score','mean_train_score'])
plt.title('fine-tuning-estimators')
plt.xlabel('estimators')
plt.ylabel('std-score')
plt.ylim(0.9,1.05)
plt.xticks([i for i in range(0,100,5)])
plt.show()

plt.figure(figsize=(12,8))
plt.plot(std_test_score)
plt.plot(std_train_score)
plt.legend(['std_test_score','std_train_score'])
plt.title('fine-tuning-estimators')
plt.xlabel('estimators')
plt.ylabel('mean-score')
plt.xticks([i for i in range(0,100,5)])
plt.show()