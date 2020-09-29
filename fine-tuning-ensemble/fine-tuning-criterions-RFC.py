#-*- codeing=utf-8 -*-
#@time: 2020/9/29 15:06
#@Author: Shang-gang Lee
#!/usr/bin/env Python

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import pandas as pd

data,label=load_iris(return_X_y=True)
RFC=RandomForestClassifier(n_estimators=9)
param={'criterion':['gini','entropy']}
clf=GridSearchCV(estimator=RFC,param_grid=param,cv=5,scoring='accuracy',n_jobs=-1,return_train_score=True)
clf.fit(data,label)
print("best_params:",clf.best_params_)
print('best_score:',clf.best_score_)
print(clf.cv_results_)
results = pd.DataFrame(clf.cv_results_)

results.to_csv('criterions-result.xls',index=False)

#visdom
criterions_result=pd.read_csv('criterions-result.xls')

mean_test_score=criterions_result['mean_test_score']
std_test_score=criterions_result['std_test_score']
mean_train_score=criterions_result['mean_train_score']
std_train_score=criterions_result['std_train_score']

plt.figure(figsize=(12,8))
plt.plot(mean_test_score)
plt.plot(mean_train_score)
plt.legend(['mean_test_score','mean_train_score'])
plt.title('fine-tuning-criterions')
plt.xlabel('criterions')
plt.ylabel('std-score')
plt.xticks(['gini','entropy'])
plt.show()

plt.figure(figsize=(12,8))
plt.plot(std_test_score)
plt.plot(std_train_score)
plt.legend(['std_test_score','std_train_score'])
plt.title('fine-tuning-criterions')
plt.xlabel('criterions')
plt.ylabel('mean-score')
plt.xticks(['gini','entropy'])
plt.show()