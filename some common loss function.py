#-*- codeing=utf-8 -*-
#@time: 2020/7/24 9:26
#@Author: Shang-gang Lee

import numpy as np
class loss_function():

    def MSE(self,pred,label):
        if len(pred)!=len(label):
            return print('plase input correct shape!')
        n=len(label)
        loss=1/n*((np.sum((pred-label)**2)))
        return loss

    def MAE(self,pred,label):
        if len(pred)!=len(label):
            return print('plase input correct shape!')
        n=len(label)
        loss=1/n*(np.sum(np.abs(pred-label)))
        return loss

    def Huber(self,a,pred,label):
        if len(pred)!=len(label):
            return print('plase input correct shape!')
        n = len(label)
        loss=0
        for i in range(n):
            if abs(pred[i]-label[i])<=a:
                loss+=1/2*((pred[i]-label[i])**2)
            else:
                loss+=a*np.abs((pred[i]-label[i]))-1/2*a**2
        return loss

    def  Cross_Entropy_Error(self,p,q):
        if len(p)!=len(q):
            return print('plase input correct shape!')
        # multi classification
        return -sum([p[i]*np.log2(q[i]) for i in range(len(p))])

def main():
    pred_y=np.array([1,6,3,8,2,0])
    label=np.array([1,5,4,8,5,1])

    #MSE loss function
    MSE_loss_func=loss_function()
    MSE_loss=MSE_loss_func.MSE(pred_y,label)
    print('MSE_loss:',MSE_loss)

    #MAE loss function
    MAE_lossfunc=loss_function()
    MAE_loss=MAE_lossfunc.MAE(pred_y,label)
    print('MAE_loss:',MAE_loss)

    #Huber loss function
    Huber_lossfunc=loss_function()
    Huber_loss=Huber_lossfunc.Huber(0.9,pred_y,label)
    print('Huber_loss:',Huber_loss)

    p = np.asarray([0.65, 0.25, 0.07, 0.03])
    q = np.array([0.6, 0.25, 0.1, 0.05])
    #Cross_Entropy_loss function
    Cross_Entropy_lossfunc=loss_function()
    Cross_Entropy_loss=Cross_Entropy_lossfunc.Cross_Entropy_Error(p,q)
    print('Cross_Entropy_loss:',Cross_Entropy_loss)

if __name__ == '__main__':
    main()

 
