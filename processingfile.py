# -*- coding: utf-8 -*-
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





from keras.models import load_model

model = load_model('fawtransferresnetcnn.h5')

img_path = 'image1111.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
print('Input image shape:', x.shape)
preds = model.predict(x)
preds *= 100
print(preds)

print('Damaged:', preds[0][0])
print('Larvae:', preds[0][1])
print('Eggs:', preds[0][2])
