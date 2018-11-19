#data preparation for the ML model 
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools


num_epoch=20



#specifying the paths to the datasets
trainpath = 'dataset/train_set'
testpath = 'dataset/test_set'
validpath = 'dataset/valid_set'


#putting the training data into batches of tensordata
train_batch = ImageDataGenerator().flow_from_directory(trainpath, 
                                                       target_size = (224, 224), 
                                                       classes = ['damaged_3_4', 'damaged_4_6', 'larvae'],  
                                                       batch_size = 5)
#putting the test data into batches of tensordata
test_batch = ImageDataGenerator().flow_from_directory(testpath, 
                                                       target_size = (224, 224), 
                                                       classes = ['damaged_3_4', 'damaged_4_6', 'larvae'],  
                                                       batch_size = 5)
#putting the validation data into batches of tensordata
valid_batch = ImageDataGenerator().flow_from_directory(validpath, 
                                                       target_size = (224, 224), 
                                                       classes = ['damaged_3_4', 'damaged_4_6', 'larvae'],  
                                                       batch_size = 3)


#creating model object
model = Sequential()

#adding the first convolution layer 
model.add(Conv2D(32, (3,3), input_shape = (224, 224, 3), activation = 'relu'))

#adding second convolution layer
model.add(Conv2D(64, (3,3), activation = 'relu'))

#adding first max pooling layer 
model.add(MaxPooling2D(pool_size = (2, 2)))

#adding third convolution layer
model.add(Conv2D(64, (3,3), activation = 'relu'))

#adding fourth convolution layer
model.add(Conv2D(128, (3,3), activation = 'relu'))

#adding  second maxpooling layer
model.add(MaxPooling2D(pool_size = (2, 2)))

#adding a flattening layer 
model.add(Flatten())

#adding a Dense layer for full connection

model.add(Dense(output_dim = 128, activation = 'relu'))

#having the softmax activation on the last Dense layer
model.add(Dense(output_dim = 3, activation = 'softmax'))

#compiling the model
model.compile(optimizer = 'sgd', loss = 'categorical_crossentropy', metrics = ['accuracy'])

hist = model.fit_generator(train_batch, steps_per_epoch = 32, 
                    validation_data = valid_batch, 
                    validation_steps = 3, epochs = num_epoch, verbose = 1)


train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(num_epoch)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

#predictions = model.predict_generator(test_batch, steps = 1, verbose = 1)















