'''
This script will use a cnn to classify mnist handwritten digits

'''
#%%
from pyexpat import model
import numpy as np
import pandas as pd
import cv2
np.random.seed(123)
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from tensorflow.keras.datasets import mnist

print("Libraries successfully loaded.")

# %%
##we pick up our image data from the keras data API

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print("Loaded data")
print("Train images batch: ", X_train.shape[0])
print("Test images batch: ", X_test.shape[0])
# %%
#we plot a couple of images to see how the dataset looks like

for image in range(5):
    plt.subplot(1,5,image+1)
    plt.imshow(X_train[image], cmap='gray')
plt.show()
# %%
#we reshape our data to fit the cnn and have only a single color channel
trainX=X_train.reshape((X_train.shape[0], 28, 28, 1))
testX=X_test.reshape((X_test.shape[0], 28, 28, 1))
# %%
print(y_train)
# %%
testX=testX.astype('float32')
trainX=trainX.astype('float32')
# %%
#ensure our image arrays are on the 0-1 scale
trainX /= 255
testX /= 255 

# %%
y_train=np_utils.to_categorical(y_train,10)
y_test=np_utils.to_categorical(y_test,10)

print(y_train)
# %%
#define our model
model=Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1),activation='relu',kernel_initializer='he_uniform'))
model.add(Conv2D(32, (3, 3),activation='relu',kernel_initializer='he_uniform'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3),activation='relu',kernel_initializer='he_uniform'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(100,activation='relu',kernel_initializer='he_uniform'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))

#we compile our model here
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()

# %%
model.fit(trainX, y_train)
# %%
for i in np.random.choice(np.arange(0, len(y_test)), size = (10,)):
    	
	probs = model.predict(testX[np.newaxis, i])
	prediction = probs.argmax(axis=1)
 
	image = (testX[i] * 255).reshape((28, 28)).astype("uint8")
 
	print("Actual digit is {0}, predicted {1}".format(y_test[i], prediction[0]))
	cv2.imshow("Digit", image)
	cv2.waitKey(0)  
# %%
