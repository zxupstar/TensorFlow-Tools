# build tf.keras

from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

from matplotlib import pyplot as plt
# 载入fashionAI数据集

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 展示数据集
train_images.shape
len(train_labels)
train_labels

# 预处理 preprocess the datas
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# 转化为浮点型
train_images = train_images / 255.0

test_images = test_images / 255.0

# build modle
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),  #全连接 128个神经元
    keras.layers.Dense(10, activation='softmax')  #训练将节点分为10种，每一种节点输出一个概率，和为1
])

# 编译模型
model.compile(optimizer='adam',#优化器-这是基于模型的数据及其损失函数来更新模型的方式。
              loss='sparse_categorical_crossentropy',#损失函数-衡量训练期间模型的准确性。希望最小化此参数，以在正确的方向上“引导”模型。
              metrics=['accuracy'])#指标-用于监视培训和测试步骤。一般使用准确性（正确分类的图像分数）。

# 训练模型
model.fit(train_images, train_labels, epochs=10)

#评估准确度
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

# 做出预测
predictions = model.predict(test_images)

# 查看预测标签
predictions[0]

### 
# 预测相关的小操作
# np.argmax(predictions[0]) 查看可能性最大的测试样本
# test_labels[0]            查看可能性最大的测试样本的真实答案
###

# 图解列出十个样本的可能性
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap='gray_r')

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

"""
查看其中一个样本的预测方法
以0为例
预测图，预测数组。正确的预测是蓝色，错误的是红色

i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()
"""
#显示所有样本的预测
# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

# 使用预测模型的方法
# Grab an image from the test dataset.
img = test_images[1]

print(img.shape)

# 即使只有一张图片也要把他加到队列之中Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))

print(img.shape)

# 开始预测
predictions_single = model.predict(img)

print(predictions_single)

plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)

#输出预测结果

np.argmax(predictions_single[0])

model.save('my_model.h5')