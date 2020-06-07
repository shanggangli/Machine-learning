# -CNN--LeNet-
LeNet分为卷积层块和全连接层块两个部分。
卷积层有2层：
第一层将数据(1,28,28)——>(6,14,14) 其中 in_channels(输入通道)=1,out_channels(输出)=6,kernel_size(卷积层size)=5

      池化层(取最大)：kernel_size=2,stride=2
      
第二层将数据(6,14,14)——>(16,4,4)in_channels=6,out_channels=16,kernel_size=5，没有补充数据，步幅=1

     池化层(取最大):kernel_size=2,stride=2
     
然后将数据扔进全连接层，硬train一发。

其中激活函数都是Sigmoid函数，但是最后trian出来的结果不是很理想，test accuracy: 0.60左右

最后将激活函数改成为 LeRU函数，test accuracy: 0.96 左右

以下是trian的结果：
Epoch:  0 | train loss: 2.3058 | test accuracy: 0.25

Epoch:  0 | train loss: 0.1556 | test accuracy: 0.92

Epoch:  0 | train loss: 0.0992 | test accuracy: 0.95

Epoch:  0 | train loss: 0.0998 | test accuracy: 0.96

Epoch:  0 | train loss: 0.0570 | test accuracy: 0.96
