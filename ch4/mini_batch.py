import sys, os
sys.path.append(os.pardir)
import numpy as np
from ch4.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize = True, one_hot_label = True)

print(x_train.shape)  # (60000, 784)
print(t_train.shape)  # (60000, 10)

train_size = x_train.shape[0]  # 랜덤샘플링을 위해 전체 개수를 가져옴(=60000)
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)  # 무작위로 뽑힌 숫자를 인덱스으로 활용
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]




