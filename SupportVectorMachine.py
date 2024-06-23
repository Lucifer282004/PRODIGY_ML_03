!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/

!kaggle datasets download -d salader/dogs-vs-cats

import zipfile
zfile = zipfile.ZipFile('/content/dogs-vs-cats.zip','r')
zfile.extractall('/content')
zfile.close()

import zipfile
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten,BatchNormalization,Dropout

# Keras Generator has been used

# For training
train_ds = keras.utils.image_dataset_from_directory(
    directory = '/content/train',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(256,256)# we want same size
)

# For validation
validation_ds = keras.utils.image_dataset_from_directory(
    directory = '/content/test',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(256,256)# we want same size
)

# Normalizing the pixel size from 0 to 255 to 0 to 1

def process(image,label):
  image = tf.cast(image/255. , tf.float32)
  return image,label

train_ds = train_ds.map(process)
validation_ds = validation_ds.map(process)

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), padding='valid', activation='relu', input_shape=(256, 256, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))

model.add(Conv2D(64, kernel_size=(3, 3), padding='valid', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))

model.add(Conv2D(128, kernel_size=(3, 3), padding='valid', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

history = model.fit(train_ds,epochs=10,validation_data=validation_ds)

import matplotlib.pyplot as pt

pt.plot(history.history['accuracy'], color='green',label='train')
pt.plot(history.history['val_accuracy'],color='blue',label='validation')
pt.legend()
pt.show()
#Before Dropout and BatchNormalization technique

import matplotlib.pyplot as pt

pt.plot(history.history['accuracy'], color='green',label='train')
pt.plot(history.history['val_accuracy'],color='blue',label='validation')
pt.legend()
pt.show()
#Graph after Dropout and BatchNormalization technique

# Here overfitting is happening

pt.plot(history.history['loss'],color='green',label='train')
pt.plot(history.history['val_loss'],color='blue',label='validation')
pt.legend()
pt.show()
#Before Dropout and BatchNormalization technique

pt.plot(history.history['loss'],color='green',label='train')
pt.plot(history.history['val_loss'],color='blue',label='validation')
pt.legend()
pt.show()
#Graph after Dropout and BatchNormalization technique

import cv2
# importing cat image
Img=cv2.imread('/content/Dog.jpg')
pt.imshow(Img)

# Resizing it to required Size
Img=cv2.resize(Img,(256,256))

# Deviding image to Batches
Img=Img.reshape((1,256,256,3))

# 0 stands for Cat
# 1 Stands for Dog

model.predict(Img)
# importing cat image
Img=cv2.imread('/content/Cat.jpg')
pt.imshow(Img)

# Resizing it to required Size
Img=cv2.resize(Img,(256,256))

# Deviding image to Batches
Img=Img.reshape((1,256,256,3))

# 0 stands for Cat
# 1 Stands for Dog
model.predict(Img)
