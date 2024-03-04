# 튜토리얼 URL {https://www.tensorflow.org/tutorials/keras/classification?hl=ko}
#TensorFlow and tf.keras
#라이브러리 임포트
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)


#M-nist 데이터 다운
fashion_mnist = tf.keras.datasets.fashion_mnist

#데이터 분류
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#데이터 탐색
print(train_images.shape)
#크기28by28 60000개의 이미지 데이터

#테스트 및 훈련 데이터 확인(시각화)
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()



#256픽셀을 지닌 데이터 정규화(0~1 사이의 값으로 출력되게..)
train_images = train_images / 255.0
test_images = test_images / 255.0
#분류 목록 확인
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()