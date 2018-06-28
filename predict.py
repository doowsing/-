import cv2                
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image
import os
if not os.path.exists("dog_name.txt"):
    print("没有狗狗品种名称！")
    quit()
with open("dog_name.txt") as f:
    dog_names = f.read().split("\n\n")
    #dog_names= [name for name in f.readlines()]
    #print(dog_names)

def path_to_tensor(img_path):
    # 用PIL加载RGB图像为PIL.Image.Image类型
    img = image.load_img(img_path, target_size=(224, 224))
    # 将PIL.Image.Image类型转化为格式为(224, 224, 3)的3维张量
    x = image.img_to_array(img)
    # 将3维张量转化为格式为(1, 224, 224, 3)的4维张量并返回
    return np.expand_dims(x, axis=0)


### 使用keras已训练好的resnet50模型分辨图片中是否有狗
from keras.applications.resnet50 import preprocess_input, decode_predictions,ResNet50
def ResNet50_predict_labels(img_path):
    # 返回img_path路径的图像的预测向量
    ResNet50_model = ResNet50(weights='imagenet')
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 


### 使用opencv库分辨图片中人脸的出现次数，并返回是否出现人脸
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

### 使用训练好的基于VGG19图像模型的迁移学习模型，返回可能性最大的狗狗品种名称
from keras.applications.vgg19 import VGG19, preprocess_input

## 加载VGG19原模型，运行时需要先下载VGG19模型
def extract_VGG19(tensor):
    identify_model=VGG19(weights='imagenet', include_top=False)
    return identify_model.predict(preprocess_input(tensor))

## 定义迁移学习模型
from keras.layers import GlobalAveragePooling2D,Dropout, Dense
from keras.models import Sequential
VGG19_model = Sequential()
VGG19_model.add(GlobalAveragePooling2D(input_shape=(7, 7, 512)))
VGG19_model.add(Dense(180, activation='sigmoid'))
VGG19_model.add(Dropout(0.3))
VGG19_model.add(Dense(133, activation='softmax'))

## 输入图片地址，返回可能性最大的狗狗品种名称
def VGG19_predict_breed(img_path):
    # 提取bottleneck特征
    bottleneck_feature = extract_VGG19(path_to_tensor(img_path))
    # 获取预测向量
    predicted_vector = VGG19_model.predict(bottleneck_feature)
    # 返回此模型预测的狗的品种
    return dog_names[np.argmax(predicted_vector)]



### 输入图片地址，打印分辨结果
def identify(img_path):
    if not os.path.exists(img_path):
        print("没有找到图片，请重新检查图片地址！")
        return

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()
    if dog_detector(img_path):
        print("The dog looks like a",VGG19_predict_breed(img_path))
    elif face_detector(img_path):
        print("Oh no,i can't imagine that you look like a ",VGG19_predict_breed(img_path))
    else:
        print("the image don't include human or dog!")

while(True):
    path = input("请输入图片地址：")
    identify(path)