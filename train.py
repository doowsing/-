import numpy as np
from keras.layers import GlobalAveragePooling2D,Dropout, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint 
import os

### 迁移学习的关键：载入已训练好的VGG19模型的瓶颈特征数据集
### 此数据集来源于：将数据集输入训练好的VGG19模型进行向前传播，取最后一层卷积层的输出
### 用作训练神经网络新加入层的训练输入数据，整个过程相当于固定神经网络前面的权重，只训练新加入的层的权重
if not os.path.exists('bottleneck_features/DogVGG19Data.npz'):
	print("开始下载VGG19瓶颈特征数据集。")
	import requests
	file_url = "https://s3.cn-north-1.amazonaws.com.cn/static-documents/nd101/DLND+documents/DogVGG19Data.npz"
	r = requests.get(file_url, stream=True)
	with open("bottleneck_features/DogVGG19Data.npz", "wb") as pdf:
	    for chunk in r.iter_content(chunk_size=1024):
	        if chunk:
	            pdf.write(chunk)
bottleneck_features = np.load('bottleneck_features/DogVGG19Data.npz')
train_VGG19 = bottleneck_features['train']
valid_VGG19 = bottleneck_features['valid']
test_VGG19 = bottleneck_features['test']

train_targets =np.loadtxt(open("train_targets/train_targets.csv"), delimiter=",", skiprows=0)
valid_targets =np.loadtxt(open("train_targets/valid_targets.csv"), delimiter=",", skiprows=0)
test_targets =np.loadtxt(open("train_targets/test_targets.csv"), delimiter=",", skiprows=0)

### TODO: 定义你的框架
VGG19_model = Sequential()
VGG19_model.add(GlobalAveragePooling2D(input_shape=train_VGG19.shape[1:]))
VGG19_model.add(Dense(180, activation='sigmoid'))
VGG19_model.add(Dropout(0.3))
VGG19_model.add(Dense(133, activation='softmax'))
VGG19_model.summary()



### TODO: 编译模型
VGG19_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

### TODO: 定义模型存储点，开始训练
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.VGG19.hdf5', 
                               verbose=1, save_best_only=True)

VGG19_model.fit(train_VGG19, train_targets, 
          validation_data=(valid_VGG19, valid_targets),
          epochs=20, batch_size=20, callbacks=[checkpointer], verbose=1)
VGG19_model.load_weights('saved_models/weights.best.VGG19.hdf5')
VGG19_predictions = [np.argmax(VGG19_model.predict(np.expand_dims(feature, axis=0))) for feature in test_VGG19]

# 报告测试准确率
test_accuracy = 100*np.sum(np.array(VGG19_predictions)==np.argmax(test_targets, axis=1))/len(VGG19_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
