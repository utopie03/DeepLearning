from keras.layers import Input, Dense, Flatten, Dropout, Activation
from keras.layers.normalization.batch_normalization_v1 import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
from keras.preprocessing import image
from keras.models import load_model
from keras.models import Model
import matplotlib.pyplot as plt
import glob, os, cv2, random, time
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense
from tensorflow.keras.optimizers import SGD
from keras.applications.vgg16 import VGG16


def processing_data(data_path):
    """
    数据处理
    :param data_path: 数据集路径
    :return: train, test:处理后的训练集数据、测试集数据
    """
    train_data = ImageDataGenerator(
        # 对图片的每个像素值均乘上这个放缩因子，把像素值放缩到0和1之间有利于模型的收敛
        rescale=1.0 / 225,
        # 浮点数，剪切强度（逆时针方向的剪切变换角度）
        shear_range=0.1,
        # 随机缩放的幅度，若为浮点数，则相当于[lower,upper] = [1 - zoom_range, 1+zoom_range]
        zoom_range=0.1,
        # 浮点数，图片宽度的某个比例，数据提升时图片水平偏移的幅度
        width_shift_range=0.1,
        # 浮点数，图片高度的某个比例，数据提升时图片竖直偏移的幅度
        height_shift_range=0.1,
        # 布尔值，进行随机水平翻转
        horizontal_flip=True,
        # 布尔值，进行随机竖直翻转
        vertical_flip=True,
        # 在 0 和 1 之间浮动。用作验证集的训练数据的比例
        validation_split=0.1,
    )

    # 接下来生成测试集，可以参考训练集的写法
    validation_data = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.1)

    train_generator = train_data.flow_from_directory(
        # 提供的路径下面需要有子目录
        data_path,
        # 整数元组 (height, width)，默认：(256, 256)。 所有的图像将被调整到的尺寸。
        target_size=(150, 150),
        # 一批数据的大小
        batch_size=32,
        # "categorical", "binary", "sparse", "input" 或 None 之一。
        # 默认："categorical",返回one-hot 编码标签。
        class_mode="categorical",
        # 数据子集 ("training" 或 "validation")
        subset="training",
        seed=0,
    )
    validation_generator = validation_data.flow_from_directory(
        data_path,
        target_size=(150, 150),
        batch_size=32,
        class_mode="categorical",
        subset="validation",
        seed=0,
    )

    return train_generator, validation_generator


def model(train_generator, validation_generator, save_model_path):
    vgg16_model = VGG16(
        weights="imagenet", include_top=False, input_shape=(150, 150, 3)
    )
    top_model = Sequential()
    top_model.add(Flatten(input_shape=vgg16_model.output_shape[1:]))
    top_model.add(Dense(256, activation="relu"))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(6, activation="softmax"))

    model = Sequential()
    model.add(vgg16_model)
    model.add(top_model)
    # 编译模型, 采用 compile 函数: https://keras.io/models/model/#compile
    model.compile(
        # 优化器, 主要有Adam、sgd、rmsprop等方式。
        optimizer=SGD(learning_rate=1e-3, momentum=0.9),
        # 损失函数,多分类采用 categorical_crossentropy
        loss="categorical_crossentropy",
        # 是除了损失函数值之外的特定指标, 分类问题一般都是准确率
        metrics=["accuracy"],
    )

    model.fit(
        # 一个生成器或 Sequence 对象的实例
        train_generator,
        # epochs: 整数，数据的迭代总轮数。
        epochs=10,
        # 一个epoch包含的步数,通常应该等于数据集的样本数量除以批量大小。
        steps_per_epoch=2276 // 32,
        # 验证集
        validation_data=validation_generator,
        # 在验证集上,一个epoch包含的步数,通常应该等于数据集的样本数量除以批量大小。
        validation_steps=251 // 32,
    )
    model.save(save_model_path)

    return model


def evaluate_model(validation_generator, save_model_path):
    # 加载模型
    model = load_model("models\knn.h5")
    # 获取验证集的 loss 和 accuracy
    loss, accuracy = model.evaluate(validation_generator)
    print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy * 100))


def predict(img):
    """
    加载模型和模型预测
    主要步骤:
        1.加载模型
        2.图片处理
        3.用加载的模型预测图片的类别
    :param img: PIL.Image 对象
    :return: string, 模型识别图片的类别,
            共 'cardboard','glass','metal','paper','plastic','trash' 6 个类别
    """
    # 把图片转换成为numpy数组
    img = img.resize((150, 150))
    img = image.img_to_array(img)

    # 加载模型,加载请注意 model_path 是相对路径, 与当前文件同级。
    model_path = "models\knn.h5"

    # 加载模型
    model = load_model(model_path)

    # expand_dims的作用是把img.shape转换成(1, img.shape[0], img.shape[1], img.shape[2])
    x = np.expand_dims(img, axis=0)

    # 模型预测
    y = model.predict(x)

    # 获取labels
    labels = {
        0: "cardboard",
        1: "glass",
        2: "metal",
        3: "paper",
        4: "plastic",
        5: "trash",
    }

    predict = labels[np.argmax(y)]

    # 返回图片的类别
    return predict


def display(train_generator, validation_generator):
    labels = train_generator.class_indices
    labels = dict((v, k) for k, v in labels.items())
        
    model_path = "models\knn.h5"
    model = load_model(model_path)
    
    test_x, test_y = validation_generator.__getitem__(1)

    preds = model.predict(test_y)
    plt.figure(figsize=(16, 16))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.title(
            "pred:%s / truth:%s"
            % (labels[np.argmax(preds[i])], labels[np.argmax(test_y[i])])
        )
        plt.imshow(test_x[i])


def main():
    data_path = "data\dataset"  # 数据集路径
    save_model_path = "models\knn.h5"  # 保存模型路径和名称
    # 获取数据
    train_generator, validation_generator = processing_data(data_path)
    # 创建、训练和保存模型
    model(train_generator, validation_generator, save_model_path)
    # 评估模型
    evaluate_model(validation_generator, save_model_path)
    # prediction
    display(train_generator, validation_generator)

if __name__ == "__main__":
    main()
