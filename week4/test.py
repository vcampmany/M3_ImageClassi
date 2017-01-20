# -*- coding: utf-8 -*-
"""

@author: master
nv"""

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from keras import backend as K
from keras.utils.visualize_util import plot
import numpy as np
import matplotlib.pyplot as plt


import numpy.ma as ma
def make_mosaic(imgs, nrows, ncols, border=0):
    """
    Given a set of images with all the same shape, makes a
    mosaic with nrows and ncols
    """
    nimgs = imgs.shape[0]
    imshape = imgs.shape[1:]
    
    mosaic = ma.masked_all((nrows * imshape[0] + (nrows - 1) * border,
                            ncols * imshape[1] + (ncols - 1) * border,3),
                            dtype=np.float32)
    
    paddedh = imshape[0] + border
    paddedw = imshape[1] + border
    for i in xrange(nimgs):
        row = int(np.floor(i / ncols))
        col = i % ncols
        
        mosaic[row * paddedh:row * paddedh + imshape[0],
               col * paddedw:col * paddedw + imshape[1],:] = imgs[i]
    return mosaic


#load VGG model
base_model = VGG16(weights='imagenet')
#visalize topology in an image
plot(base_model, to_file='modelVGG16.png', show_shapes=True, show_layer_names=True)

#read and process iamge
img_path = '/data/MIT/test/coast/art1130.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
plt.imshow(img)
plt.show()

#crop the model up to a certain layer
model = Model(input=base_model.input, output=base_model.get_layer('block5_conv2').output)
#get the features from images
features = model.predict(x)

# Get weights of the first layer TASK 1
weights = base_model.get_layer('block1_conv1').get_weights()

view_weights = np.zeros((24,24,3))
for i in range(weights[0].shape[3]):
	col = (i % 8)*3
	row = (i / 8)*3
	#print weights[0][:,:,:,i]
	view_weights[row:row+3,col:col+3,:] = weights[0][:,:,:,i]
	#plt.imshow(weights[0][:,:,:,i],interpolation='nearest')
	#plt.show()
	
view_weights = np.rollaxis(weights[0],3)
view_weights = make_mosaic(view_weights, 8,8)
plt.imshow(view_weights,interpolation='nearest')
plt.show()

# Get activations of a 3rd block layer TASK 2
model_t2 = Model(input=base_model.input, output=base_model.get_layer('block3_conv2').output)
features_t2 = model_t2.predict(x)


#for i in range(features_t2.shape[3]):
#	plt.imshow(features_t2[0,:,:,i],interpolation='nearest')
#	plt.show()

fig = plt.figure()
fig.suptitle('Mean activation map')
features_mean = np.mean(features_t2[0,:,:,:], axis=2)
plt.imshow(features_mean,interpolation='nearest')
plt.show()

fig = plt.figure()
fig.suptitle('Max activation map')
features_max = np.max(features_t2[0,:,:,:], axis=2)
plt.imshow(features_max,interpolation='nearest')
plt.show()
	
	
	
	
	
	
	

