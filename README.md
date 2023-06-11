# cnn-handwriting-recognition

#### 介绍
使用卷积神经网络模型识别手写数字 
> 学习《深度学习入门》一书后，整理代码添加注释方便后续使用

#### 目录说明
- main 模型结构等核心代码
- common 模型用到公用功能
- dataset 训练用数据集

#### 如何运行
- 安装python
- 使用 MNIST 手写数字图像集，网址http://yann.lecun.com/exdb/mnist/
- 如果需要训练数据，可以运行train_deepnet.py，会自动下载训练数据
- 如果需要统计准确率以及查看没有识别的图片，可以运行misclassified_mnist.py
- 如果需要预测数据，运行awesome_net.py，里面会随机取一张mnist的测试图片，关闭图片后会打印出预测数字


