import tensorflow as tf

# 载入数据集 
mnist = tf.keras.datasets.mnist

# 将样本从整数转换为浮点数
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 将模型各层堆叠起来，以搭建tf.keras.Sequential模型。为训练选择损失函数
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练并验证模型
model.fit(x_train, y_train, epochs=50)
model.evaluate(x_test, y_test)

 