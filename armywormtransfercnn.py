#data preparation for the ML model 

import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Activation
from keras.layers.core import Dense, Flatten, Dropout
from keras import optimizers
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.engine.topology import get_source_inputs

num_epoch=15
num_classes = 3



#specifying the paths to the datasets
trainpath = 'prime_data/train_set'
validpath = 'prime_data/valid_set'



traingen = ImageDataGenerator(rescale = 1./255,
                              shear_range = 0.2,
                              zoom_range = 0.2,
                              horizontal_flip = True)


#putting the training data into batches of tensordata
train_batch = traingen.flow_from_directory(trainpath, 
                                           target_size = (224, 224), 
                                           classes = ['damaged','larvae','eggs'],  
                                           batch_size = 10)

validgen = ImageDataGenerator(rescale = 1./255,
                              shear_range = 0.2,
                              zoom_range = 0.2,
                              horizontal_flip = True)


#putting the validation data into batches of tensordata
valid_batch = validgen.flow_from_directory(validpath, 
                                           target_size = (224, 224), 
                                           classes = ['damaged','larvae','eggs'],  
                                           batch_size = 4)

'''
#when training our model we would want to have the same random numbers generated for  the weights on every training 
#we therefore set a constant random seed for numpy, python and the backend engine TensorFlow
import random as rn
import tensorflow as tf
import numpy as np

import os
os.environ['PYTHONHASHSEED'] = '0'

#setting costant random seed number for numpy
np.random.seed(23)

#setting costant random seed number for python
rn.seed(24)

#setting costant random seed number for tensorflow
tf.set_random_seed(25)

#we need to force tensorflow to runa single thread 
from keras import backend as K

sess = tf.Session(graph = tf.get_default_graph(), config=tf.session_conf)
K.set_session(sess)
'''


#we can use transfer learning on any of these models
'''
#Load the Inception_V3 model
inception_model = keras.applications.inception_v3.InceptionV3(weights='imagenet')
'''
#Load the ResNet50 model
resnet_model = keras.applications.resnet50.ResNet50(weights='imagenet')

'''
#Load the MobileNet model
mobilenet_model = keras.applications.mobilenet.MobileNet(weights='imagenet')
'''
#printing the model summary to see the layers
resnet_model.summary()
'''

#importing the vgg16 from the keras packages
vgg16model = keras.applications.vgg16.VGG16()

#printing the model summary to see the layers
mobilenet_model.summary()
'''
#printing the model type
print(type(resnet_model))

#taking all layers in mobile net apart frm the last one and making them outputs
x = resnet_model.layers[-1].output

#creating the last layer of our new model
last_layer = Dense(output_dim = num_classes, activation = 'softmax')(x)

#creating our model and adding up the layers tht has 3 classes
model = Model(input=resnet_model.input, output= last_layer)

'''
#changing the model type from model to squential 
model = Sequential()
for layer in resnet_model.layers:
    model.add(layer)
    
#seeing the summary of the new model
model.summary()

#removing the last dense layer
model.layers.pop()
'''
#seeing the summary of the model
model.summary()



#freezing the weights of the other layers apart from the last 23 so that they aren't changed during training
for layer in model.layers[:-23]:
    layer.trainable = False
    
 
model.compile(optimizer = optimizers.Adam(lr=0.0001), loss = 'categorical_crossentropy', metrics = ['accuracy'])

hist = model.fit_generator(train_batch, steps_per_epoch = 32, 
                    validation_data = valid_batch, 
                    validation_steps = 4, epochs = num_epoch, 
                    shuffle = True, 
                    verbose = 1)


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




model.save('fawtransfermobilecnn.h5')


from keras.models import load_model

new_model = load_model('fawtransfermobilecnn.h5')

#showing the summary of the model
new_model.summary()


img_path = 'test2.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
print('Input image shape:', x.shape)
preds = new_model.predict(x)
preds *= 100
print(preds)

print('Damaged:', preds[0][0])
print('Larvae:', preds[0][1])
print('Eggs:', preds[0][2])

#building Python Dictionary with results
results = {          
           'Damaged': preds[0][0],
           'Larvae' : preds[0][1],
           'Eggs'   : preds[0][2],
          }















