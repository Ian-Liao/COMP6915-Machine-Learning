"""Assignment 4
### Import Libraries
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import seaborn as sns
import tensorflow as tf

from scipy.io import loadmat

from skimage import color
from skimage import io
from sklearn.model_selection import train_test_split

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten

#Importing the CNN related layers 
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.regularizers import l1
from keras.regularizers import l2


# user-defined functions

def load_data(path):
    """ Helper function for loading a MAT-File"""
    data = loadmat(path)
    return data['X'], data['y']



def plot_images(img, labels, nrows, ncols):
    """ a helper function to display images . 
    Note that color and gray picture need different codes:
    """
    fig, axes = plt.subplots(nrows, ncols)
    for i, ax in enumerate(axes.flat): 
        if img[i].shape == (32, 32, 3):
          # for color
            ax.imshow(img[i]) 
        else:
          #for grayscale
            ax.imshow(img[i,:,:,0])
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(labels[i])
        
        
        
def plot_prediction_image(i, predictions_array, true_label, img):
    """  plot predicton and the true label for images"""
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])  
    plt.imshow(img[:,:,0], cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
      
    # Color correct predictions in blue, incorrect predictions in red
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
      
    plt.xlabel("predicted:{} with {:2.0f}% (real:{})".format(predicted_label,
    100*np.max(predictions_array),true_label),color=color)



def plot_graph(history):
    """ Plotting the accuracy and loss for different epochs"""
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title( 'model accuracy')
    plt.ylabel( 'accuracy' )
    plt.xlabel( 'epoch' )
    plt.legend([ 'train' , 'test' ], loc= 'upper left' )
    plt.show()
    # summarize history for loss
    plt.plot(history.history[ 'loss' ])
    plt.plot(history.history[ 'val_loss' ])
    plt.title( 'model loss')
    plt.ylabel( 'loss' )
    plt.xlabel( 'epoch' )
    plt.legend([ 'train' , 'test' ], loc= 'upper left' )
    plt.show()



# %matplotlib inline
""" increase the defualt size of the images"""
plt.rcParams['figure.figsize'] = (20.0, 5.0)

""" fix a random seed for reproducibility"""
seed = 0
np.random.seed(seed)


""" importing the data"""
# make sure the .mat files are in the same folder as .py file
X_train, y_train = load_data("train_32x32.mat")
X_test, y_test = load_data("test_32x32.mat")


# check data shape
print("Training Set", X_train.shape, y_train.shape)
print("Test Set", X_test.shape, y_test.shape)

"""**Transposing the the train and test data by converting it from
  (width, height, channels, size) -> (size, width, height, channels)**
"""

# Only transpose the image arrays if the channel is not the last column
if (X_train.shape[3] != 3):
  X_train, y_train = X_train.transpose((3,0,1,2)), y_train[:,0]
  X_test, y_test = X_test.transpose((3,0,1,2)), y_test[:,0]

# check new data shape
print("Training Set", X_train.shape)
print("Test Set", X_test.shape)

# Calculate the total number of images
num_images = X_train.shape[0] + X_test.shape[0]

print("Total Number of Images", num_images)


"""Plot some of the images:"""

print("y_train is",y_train)
plot_images(X_train, y_train,2,10)

# Make grayscale
# if channel is 3, then it is RGB. convert it to grayscale
if X_train.shape[3] == 3:
  X_train= np.dot(X_train,[0.30,0.59,.11])
  X_train = np.expand_dims(X_train,axis=3).astype('float32')
  X_train.shape
  
if X_test.shape[3] == 3:
  X_test= np.dot(X_test,[0.30,0.59,.11])
  X_test = np.expand_dims(X_test,axis=3).astype('float32')
  X_test.shape

"""### print some of the test data"""

plot_images(X_test, y_test, 1, 10)

"""### Normalize the data"""

# scale the data:

X_train_norm = X_train/255
X_test_norm = X_test/255

"""
Use Keras to make the label categorical
"""
y_train[y_train==10] = 0
y_test[y_test==10] = 0
y_train_cat = np_utils.to_categorical(y_train)
y_test_cat = np_utils.to_categorical(y_test)
num_classes = y_train_cat.shape[1]
print("the number of categorical classess is:",num_classes)

"""### Getting the number, height and width of the images:"""

training_samples, height, width, channel = X_train_norm.shape
testing_samples,_,_,_  = X_test_norm.shape

"""### The model:"""

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(height,width ,channel), activation='relu'))
model.add(Conv2D(64, (5,5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.20))
model.add(Flatten())
model.add(Dense(128, activation='relu', activity_regularizer=l2(0.0025)))
model.add(Dense(num_classes, activation='softmax'))


"""Compile model"""
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

"""**Fit the model**"""
history = model.fit(X_train_norm, y_train_cat, validation_data=(X_test_norm, y_test_cat),epochs=1, batch_size=100)

""" Show a summary of the model"""
model.summary()

"""calculate the scores and loss function for test data"""
score = model.evaluate(X_test_norm, y_test_cat, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


"""make predictions"""

predictions = model.predict(X_test_norm)


"""Plot the six first X test images, their predicted label, and the true label"""
num_rows = 1
num_cols = 6
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_prediction_image(i, predictions, y_test, X_test)
plt.show()

old_history= history

""" plot the accuracy and loss for epochs"""
plot_graph(history)

"""Save entire model to a HDF5 file"""
model.save('my_model.h5')

