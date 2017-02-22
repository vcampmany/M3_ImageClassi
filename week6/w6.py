from keras.preprocessing import image
from keras import backend as K
from keras.utils.visualize_util import plot
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import argparse
import os
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam

from keras.layers.noise import GaussianNoise, GaussianDropout
from keras.models import Sequential, Model
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Input

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
parser.add_argument('-batch_size', default=64, type=int)
parser.add_argument('-n_epochs', default=30, type=int)
parser.add_argument('-opt', help='Optimizer to use', type=str, default='Adam')
parser.add_argument('-lr', default=0.000001, type=float)
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
img_width = 128
img_height = 128
batch_size=args.batch_size
number_of_epoch=args.n_epochs

# set Training folder
train_data_dir='/home/master/M3_ImageClassi/MIT_split/'+args.train_folder

# DEFINE MODEL
in_ = Input(shape=(img_width, img_height, 3))
#x = GaussianNoise(10)(in_)
x = Convolution2D(32, 5, 5)(in_)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = BatchNormalization()(x)

x = Convolution2D(64, 5, 5)(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = BatchNormalization()(x)

#x = Convolution2D(128, 3, 3)(x)
#x = BatchNormalization()(x)
#x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Flatten()(x) # this converts our 3D feature maps to 1D feature vectors
x = Dense(512, W_regularizer=l2(weight_decay))(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)
x = Dense(512, W_regularizer=l2(weight_decay))(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)
x = Dense(8)(x)
x = Activation('softmax')(x)

model = Model(input=in_, output=x)
mname = 'prova23'
plot(model, to_file=mname+'.png', show_shapes=True, show_layer_names=True)
# END DEFINE MODEL

if args.opt == 'SGD':
  optimizer = SGD(lr=args.lr, momentum=args.momentum, decay=args.lr_decay)
elif args.opt == 'SGDNesterov':
  optimizer = SGD(lr=args.lr, momentum=args.momentum, decay=args.lr_decay, nesterov=True)
elif args.opt == 'RMSprop':
  optimizer = RMSprop(lr=args.lr, rho=0.9, epsilon=1e-08, decay=0.0)
elif args.opt == 'Adagrad':
  optimizer = Adagrad(lr=args.lr, epsilon=1e-08, decay=0.0)
elif args.opt == 'Adadelta':
  optimizer = Adadelta(rho=0.95, epsilon=1e-08, decay=0.0)
elif args.opt == 'Adam':
  optimizer = Adam(lr=args.lr)

model.compile(loss='categorical_crossentropy',optimizer=optimizer, metrics=['accuracy'])
for layer in model.layers:
    print layer.name, layer.trainable

model.summary()

datagen = ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    preprocessing_function = preprocess_input,
    rotation_range=args.rotation,
    width_shift_range=args.w_shift,
    height_shift_range=args.h_shift,
    shear_range=0.,
    zoom_range=args.zoom,
    channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=args.h_flip,
    vertical_flip=False,
    rescale=None)

datagen_orig = ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    preprocessing_function = preprocess_input,
    rotation_range=args.rotation,
    width_shift_range=args.w_shift,
    height_shift_range=args.h_shift,
    shear_range=0.,
    zoom_range=args.zoom,
    channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=args.h_flip,
    vertical_flip=False,
    rescale=None)
    
train_generator = datagen.flow_from_directory(train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

test_generator = datagen_orig.flow_from_directory(test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

validation_generator = datagen_orig.flow_from_directory(val_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

history=model.fit_generator(train_generator,
        samples_per_epoch=count_samples(train_data_dir)*3,
        nb_epoch=number_of_epoch,
        validation_data=validation_generator,
        nb_val_samples=count_samples(val_data_dir))


result = model.evaluate_generator(test_generator, val_samples=count_samples(test_data_dir))

# Print the output of the evaluate generator
for i in range(len(result)):
  print 'test', model.metrics_names[i], result[i]
#print result

model.save_weights('baseline_weights.dat')

# list all data in history

if True:
  # summarize history for accuracy
  plt.plot(history.history['acc'])
  plt.plot(history.history['val_acc'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'validation'], loc='upper left')
  plt.savefig(mname+'_accuracy.jpg')
  plt.close()
  # summarize history for loss
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'validation'], loc='upper left')
  plt.savefig(mname+'_loss.jpg')
