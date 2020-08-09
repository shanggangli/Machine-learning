#-*- codeing=utf-8 -*-
#@time: 2020/8/9 14:09
#@Author: Shang-gang Lee

import numpy as np
import matplotlib.pyplot as plt
train_data=np.array([(1,2),(1,3),(2,2),(2,0),(0,2),(4,4),(5,5),(4,7),(6,5),(8,7)])
train_label=np.array([-1,-1,-1,-1,-1,1,1,1,1,1])
test_data=np.array([(0,0),(4,5),(1,1),(6,8)])
class perceptron:
    def __init__(self,w,b):
        self.w=w
        self.b=b

    def Predtion(self,X):
        y=[]
        self.X=X
        for i in X:
            y.append((np.dot(self.w,i)+self.b))
        return y

    def loss(self,label,pred_y):
        n=0.01
        Loss=0
        for i in range(len(label)):
            if label[i]*pred_y[i]<=0:
                Loss+=-label[i]*pred_y[i]

                self.w=self.w+n*label[i]*self.X[i]
                self.b=self.b+n*label[i]
        return Loss,self.w,self.b

    def sign(self,test_data):
        Y=[]
        y=self.Predtion(test_data)
        for i in y:
            Y.append((1 if i >= 0 else -1))
        return Y

    def train(self,train_data,label):
        for i in range(100):
            y=self.Predtion(train_data)
            loss,self.w,self.b=self.loss(label,y)
            print('step:',i,'|  loss',loss)
            if loss==0 and self.w[1]!=0:
                break

        return self.w,self.b
if __name__ == '__main__':
    # train data
    x=[i[0] for i in train_data]
    y=[i[1] for i in train_data]
    scatter=plt.scatter(x,y)
    plt.legend()
    # test data
    x1=[i[0] for i in test_data]
    y1=[i[1] for i in test_data]
    scatter1=plt.scatter(x1,y1)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend((scatter,scatter1),('train data',('test data')),loc='upper left')
    plt.show()

    Perceptron=perceptron([0.1,0.5],0.5)
    w,b=Perceptron.train(train_data,train_label)
    print('X1:%.2f   '%w[0],'X2:%.2f   '%w[1],'b:%.2f   '%b)
    print(test_data)
    print(Perceptron.sign(test_data))

    X=x+x1
    Y=y+y1
    scatter2=plt.scatter(X,Y)
    X_line=np.array([1,2,3,4,5,6,7,8,9,10])
    W=-(w[0]/w[1])
    B=-(b/w[1])
    print('斜率:%.2f   '%W,'截距:%.2f   '%B)
    Y_line=W*X_line+B
    plt.plot(X_line,Y_line, color='red')
    plt.ylim(-1,10)
    plt.xlim(-1,10)
    plt.show()
