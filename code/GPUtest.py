import tensorflow as tf

# 测试 GPU 是否可用GPU
print("--------------------------------------")
print(tf.__version__)
print("--------------------------------------")
print(tf.test.gpu_device_name())
print("--------------------------------------")
print(tf.config.experimental.set_visible_devices)
print("--------------------------------------")
print("GPU:", tf.config.list_physical_devices("GPU"))
print("--------------------------------------")
print("CPU:", tf.config.list_physical_devices(device_type="CPU"))
print("--------------------------------------")
# 输出可用的GPU数量
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))


# 查询GPU设备
# mnist = tf.keras.datasets.mnist
#
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0
#
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(input_shape=(28, 28)),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(10, activation='softmax')
# ])
#
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#
# model.fit(x_train, y_train, epochs=5)
#
# model.evaluate(x_test, y_test, verbose=2)
# 正常来说这段demo用gpu的话十几秒就能训练好，证明是可以使用gpu进行推理和反传的。