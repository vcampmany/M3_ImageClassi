from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras import backend as K
from keras.utils.visualize_util import plot
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import argparse
import os
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam
import matplotlib.pyplot as plt
import scipy.misc

def preprocess_input(x, dim_ordering='default'):
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    assert dim_ordering in {'tf', 'th'}

    if dim_ordering == 'th':
        # 'RGB'->'BGR'
        x = x[ ::-1, :, :]
        # Zero-center by mean pixel
        x[ 0, :, :] -= 103.939
        x[ 1, :, :] -= 116.779
        x[ 2, :, :] -= 123.68
    else:
        # 'RGB'->'BGR'
        x = x[:, :, ::-1]
        # Zero-center by mean pixel
        x[:, :, 0] -= 103.939
        x[:, :, 1] -= 116.779
        x[:, :, 2] -= 123.68
    return x

def count_samples(train_dir):
  return sum([len(files) for r, d, files in os.walk(train_dir)])

parser = argparse.ArgumentParser()
parser.add_argument('-test', help='Number of test to use', type=str, default='t0')
parser.add_argument('-train_folder', help='Train folder to use', type=str, default='train')
parser.add_argument('--batch_norm', action='store_true')
parser.add_argument('-drop', default=0.0, type=float)
parser.add_argument('-wd', default=0.0, type=float)
parser.add_argument('-batch_size', default=32, type=int)
parser.add_argument('-n_epochs', default=20, type=int)
parser.add_argument('-opt', help='Optimizer to use', type=str, default='Adadelta')
parser.add_argument('-lr', default=0.001, type=float)
parser.add_argument('-momentum', default=0.9, type=float)
parser.add_argument('-act', help='Activations to use', type=str, default='relu')
parser.add_argument('--h_flip', action='store_true')
parser.add_argument('-zoom', default=0.0, type=float)
parser.add_argument('-w_shift', default=0.0, type=float)
parser.add_argument('-h_shift', default=0.0, type=float)
parser.add_argument('-rotation', default=0.0, type=float)
parser.add_argument('-lr_decay', default=0.0, type=float)

args = parser.parse_args()
test = args.test
batch_norm = args.batch_norm
dropout = args.drop
weight_decay = args.wd

print(args)

val_data_dir='/home/master/data/MIT/validation'
test_data_dir='/home/master/data/MIT/test'
img_width = 224
img_height = 224
batch_size=args.batch_size
number_of_epoch=args.n_epochs

# set Training folder
train_data_dir='/home/master/M3_ImageClassi/MIT_split/'+args.train_folder

datagen = ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    preprocessing_function = None,
    rotation_range=0.0,
    width_shift_range=0.0,
    height_shift_range=0.0,
    shear_range=0.,
    zoom_range=0.1,
    channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None)

train_generator = datagen.flow_from_directory(train_data_dir,
        target_size=(img_width, img_height),
        batch_size=1,
        class_mode='categorical')

for X_batch, y_batch in train_generator:
  #pyplot.imshow(X_batch[0].reshape(28, 28), cmap=pyplot.get_cmap('gray'))
  print(X_batch[0].shape)
  scipy.misc.imsave('dataug_samples/zoom.jpg', X_batch[0])
  break