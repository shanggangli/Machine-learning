#-*- codeing=utf-8 -*-
#@time: 2020/8/10 11:19
#@Author: Shang-gang Lee

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

iris=load_iris() #load data
df=pd.DataFrame(iris.data)
df['label']=iris.target
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
X_train,X_test, y_train, y_test=train_test_split(df.iloc[:,:4],df.iloc[:,-1],random_state=None,train_size=0.8)
print("X_train:",X_train.shape,
      "X_test:",X_test.shape,
      "y_train",y_train.shape,
      "y_test:",y_test.shape)

class Kmeans:
    def __init__(self,K):
        self.K=K

    def train(self,train_data):
        train_data=np.array(train_data)
        label=np.zeros(len(train_data))
        Kdata = []
        for i in range(0,self.K):
            X=np.random.randint(0,len(train_data))
            Kdata.append(train_data[X])
        print("初始化的质心:",Kdata)
        dis=[]
        for n in range(500):    #update 500 times clusters
            for i in range(len(train_data)):    # Calculate the distance between each sample and the cluster
                for k in Kdata:
                    dis.append(self.Eu_distance(k,train_data[i]))
                cluster=np.argmin(dis)  #take the smallest distance index as the label of the sample
                label[i]=cluster
                dis.clear()

            for t in range(len(Kdata)):     #update clusters by the mean of distance in each label
                points=[train_data[j] for j in range(len(train_data)) if label[j]==t]
                Kdata[t]=np.mean(points,axis=0)

            for i in range(len(Kdata)):     #check out some culster's values if is NAN.
                if isinstance(Kdata[i],np.float64):     # if value==NAN is True ,we need to random new data for it.
                    for j in range(train_data.shape[1]):
                        Kdata[i]=np.random.randint(0,max(train_data[:,j]),size=self.K)
        self.C=Kdata
        print("最终的质心:",Kdata)
        return label

    def Eu_distance(self,Kdata,data):   #欧氏距离
        dis=0
        for i in range(len(Kdata)):
            dis+=np.sqrt(np.sum(np.square(Kdata[i]-data[i])))
        return dis

    def prediction(self,test_data): # prediction funntion
        test_data=np.array(test_data)
        label=np.zeros(len(test_data))
        test_dis=[]
        for i in range(len(test_data)):
            for j in self.C:
                test_dis.append(self.Eu_distance(j,test_data[i]))
            label[i]=np.argmin(test_dis)
            test_dis.clear()
        return  label

if __name__ == '__main__':
    K=3 # split to 3 clusters
    kmean=Kmeans(K)
    label=kmean.train(X_train)
    test_label=kmean.prediction(X_test)
    print('original labels:',np.array(y_train))
    print("After Kmeans:",label)

    # visual
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_r = pca.fit(X_train).transform(X_train)
    plt.subplot(131)
    plt.scatter(X_r[:,0],X_r[:,1])
    plt.title('original data')

    plt.subplot(132)
    plt.scatter(X_r[:,0],X_r[:,1],c=np.array(y_train))
    plt.title('original labels')

    plt.subplot(133)
    plt.scatter(X_r[:,0],X_r[:,1],c=label)
    plt.title('After Kmeans')
    plt.show()
