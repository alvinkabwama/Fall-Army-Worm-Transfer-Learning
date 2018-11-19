#data preparation for the ML model 
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten, Dropout
from keras.optimizers import Adam
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


num_epoch=30
num_classes = 3



#specifying the paths to the datasets
trainpath = 'prime_data/train_set'
testpath = 'prime_data/test_set'
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
'''

#putting the test data into batches of tensordata
test_batch = ImageDataGenerator().flow_from_directory(testpath, 
                                                       target_size = (224, 224), 
                                                       classes = ['damaged','larvae','eggs'],  
                                                       batch_size = 1)
'''
validgen = ImageDataGenerator(rescale = 1./255,
                              shear_range = 0.2,
                              zoom_range = 0.2,
                              horizontal_flip = True)


#putting the validation data into batches of tensordata
valid_batch = validgen.flow_from_directory(validpath, 
                                           target_size = (224, 224), 
                                           classes = ['damaged','larvae','eggs'],  
                                           batch_size = 4)


#creating model object
model = Sequential()

#adding the first convolution layer 
model.add(Conv2D(32, (3,3), input_shape = (224, 224, 3), activation = 'relu'))

#adding second convolution layer
model.add(Conv2D(64, (3,3), activation = 'relu'))

#adding first max pooling layer 
model.add(MaxPooling2D(pool_size = (2, 2)))

#adding a dropout layer
model.add(Dropout(0.20))

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

#adding another dropout 
model.add(Dropout(0.25))

#having the softmax activation on the last Dense layer
model.add(Dense(output_dim = num_classes, activation = 'softmax'))

#compiling the model
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

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



'''
model.save('armywormmodeldata3.h5')

predictions = model.predict_generator(test_batch, steps = 1, verbose = 1)

for i in predictions:
    print(i)


from keras.models import load_model

new_model = load_model('armywormmodeldata3.h5')

#showing the summary of the model
new_model.summary()

#getting the weights of the model
new_model.get_weights()



img_path = 'testimage.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
print('Input image shape:', x.shape)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds))

predictions = model.predict_generator(test_batch, steps = 1, verbose = 1)
for i in predictions:
    print(i)
    '''















