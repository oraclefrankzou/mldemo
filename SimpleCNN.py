

import tensorflow as tf
import numpy as np
import keras
import matplotlib.pyplot   as plt
from keras import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten
from tensorflow.keras.utils import to_categorical
from keras.losses import CategoricalCrossentropy
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import sys




"""
   module:  dog and cat classier from <deep learning from python>
   author:
   date

"""

        
"""
   func: jpeg genetor , pix: 150*150
   author:
   date

"""

def jpgToGeneror(datadir):

    datagenerator = ImageDataGenerator(rescale=1./225)
    result = datagenerator.flow_from_directory(
        datadir,
        target_size=(150,150),
        batch_size=20,
        class_mode='binary'
    )
    return result

"""
   func: plot the metric
   author:
   date

"""
def plotModelMetric(history):
    accuracy = history.history['accuracy']
    x = [x for x in range(1, len(accuracy) + 1)]
    acc_y = history.history['accuracy']
    valacc_y = history.history['val_accuracy']
    plt.plot(x, acc_y, 'b', label='accuracy')
    plt.plot(x, valacc_y, 'bo', label='validate_accuracy')
    plt.legend()
    plt.show()


"""
   func: read jpg from path
   author:
   date

"""
def loadImage():
    img = image.load_img('C:\\test\\dogandcat\\valid-data\\cat\\cat.1000.jpg',target_size=(150,150))
    imgarray = image.img_to_array(img)
    imgarray = np.expand_dims(imgarray,axis=0)
    imgarray = imgarray/225.
    return imgarray

  
  
"""
   func: build convolution network
   author:
   date

"""

def main():
      TRAIN_DIR = "C:\\test\dogandcat\\train-data"
      VAILD_DIR = "C:\\test\dogandcat\\valid-data"
      TEST_DIR = "C:\\test\dogandcat\\test-data"
      model = Sequential()
      model.add(Conv2D(64,(3,3),activation='relu',input_shape=(150,150,3)))
      model.add(MaxPooling2D(2,2))
      model.add(Conv2D(128,(3,3),activation='relu'))
      model.add(MaxPooling2D(2, 2))
      model.add(Conv2D(128, (3, 3), activation='relu'))
      model.add(MaxPooling2D(2, 2))
      model.add(Flatten())
      model.add(keras.layers.Dropout(0.5))
      model.add(Dense(512,activation='relu'))
      model.add(Dense(1,activation='sigmoid'))
      model.compile(optimizer='rmsprop',loss= 'binary_crossentropy',metrics=['accuracy'])
      traingenerator = jpgToGeneror(TRAIN_DIR)
      validgenerator = jpgToGeneror(VAILD_DIR)
      history = model.fit_generator(
          traingenerator,
          steps_per_epoch=100,
          epochs=10,
          validation_data=validgenerator,
          validation_steps=50
      )
      plotModelMetric(history)

if __name__ == '__main__':
    main()
