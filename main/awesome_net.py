# Create your awesome net!!
# 预测手写数字
import sys, os
current_dir = os.getcwd()  # 获取当前路径
sys.path.append(current_dir)
import random
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from deep_convnet import DeepConvNet


network = DeepConvNet()  
network.load_params(file_name="main/deep_convnet_params.pkl")

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=False, flatten=False, one_hot_label=False)
    return x_test
# print(get_data().shape)
count = random.randint(0,get_data().shape[0])
# print(count)
img = np.reshape(get_data()[count:(count+1)],(28,28))
plt.imshow(img)
plt.show()

predicted_labels = network.predict(get_data()[count:(count+1)])
max_indices = np.argmax(predicted_labels, axis=1) 

print(max_indices)