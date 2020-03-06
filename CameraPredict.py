from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
AUTOTUNE = tf.data.experimental.AUTOTUNE

import IPython.display as display
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

#print(tf.__version__)
from ecapture import ecapture as ec 

ec.capture()


model = keras.models.load_model('/home/falcon/garbageai/model/keras_model.h5')
#Retreive Images
test_image = '/home/falcon/garbageai/test_cb_plastic/tests/pic2.png'

im = load_img(test_image, target_size=(150, 150)) # -> PIL image
im_array = img_to_array(im) 
im_array = np.expand_dims(im_array, axis=0)
im_array[0] = im_array[0] / 255

prediction = model.predict(im_array)