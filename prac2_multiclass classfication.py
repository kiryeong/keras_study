# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 11:32:51 2020

@author: SAMSUNG
"""

#로이터(reuter) 뉴스를 46개의 상호 배타적인 토픽으로 분류하는 신경망

from keras.datasets import reuters
import numpy as np
from keras.utils.np_utils import to_categorical
from keras import models
from keras import layers
import matplotlib.pyplot as plt


(train_data, train_labels), (test_data, test_labels) = reuters.load_data(
    num_words = 10000)

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_train = vectorize_sequences(train_data) #훈련데이터 벡터 변환
x_test = vectorize_sequences(test_data)
'''
def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i , label in enumerate(labels):
        results[i, label] = 1.
    return results

one_hot_train_labels = to_one_hot(train_labels) #훈련 레이블 벡터 변환
one_hot_test_labels = to_one_hot(test_labels)
'''

# 레이블을 벡터로 바꾸는 방법은 1) 레이블의 리스트를 정수 텐서로 변환하는것, 2) 원-핫 인코딩을 사용하는 것 
# 여기서는 2)를 사용
# 만약 1)을 사용한다면 
# y_train = np.array(train_labels)
# y_test = np.array(test_labels)
# 를 쓴 뒤 손실함수를 sparse_categorical_crossentropy 를 쓴다. (정수 레이블을 기대할 때)
# categorical_crossentropy는 레이블이 범주형 인코딩되어 있을 것이라고 기대한다.

one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)
'''
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
'''
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]
'''
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs = 20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label = 'Training loss') #'bo'는 파란색 점을 의미
plt.plot(epochs, val_loss, 'b', label = 'Validation loss') #'b'는 파란색 실선을 의미
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
'''

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs = 9,
                    batch_size=512,
                    validation_data=(x_val, y_val))

results = model.evaluate(x_test, one_hot_test_labels)

# 단일 레이블, 다중 분류 문제에서는 N개의 클래스에 대한 확률 분포를 출력하기 위해 softmax 활성화 함수를 사용해야 한다.
# 이런 문제에는 항상 범주형 크로스엔트로피를 사용해야 한다. 이 함수는 모델이 출력한 확률분포와 타깃 분포 사이의 거리를 최소화한다.


