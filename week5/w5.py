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

val_data_dir='/home/master/data/MIT/validation'
test_data_dir='/home/master/data/MIT/test'
img_width = 224
img_height=224
batch_size=32
number_of_epoch=20

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
parser.add_argument('--dataug', action='store_true')
parser.add_argument('--dropout', action='store_true')
args = parser.parse_args()
test = args.test
dataug = args.dataug
dropout = args.dropout

# set Training folder
train_data_dir='/home/master/M3_ImageClassi/MIT_split/'+args.train_folder

# create the base pre-trained model
base_model = VGG16(weights='imagenet')
plot(base_model, to_file='modelVGG16a.png', show_shapes=True, show_layer_names=True)

x = base_model.get_layer('block5_pool').output
#x = base_model.layers[-2].output
#x = base_model.get_layer('block5_pool').output
#x = Flatten(name='flat')(x)
#x = Dense(4096, activation='relu', name='fc')(x)

if test == 't0':
    x = Flatten(name='flat')(x)
    #x = base_model.get_layer('block4_pool').output
if test =='t1':
    x = Flatten(name='flat')(x)
    x = Dense(1024, activation='relu', name='fc1')(x)
    if dropout:
      x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu', name='fc2')(x)
    if dropout:
      x = Dropout(0.5)(x)
elif test=='t2':
    x = Flatten(name='flat')(x)
    x = Dense(512, activation='relu', name='fc1')(x)
    if dropout:
      x = Dropout(0.5)(x)
    x = Dense(512, activation='relu', name='fc2')(x)
    if dropout:
      x = Dropout(0.5)(x)
elif test == 't3':
    x = Flatten(name='flat')(x)
    x = Dense(256, activation='relu', name='fc1')(x)
    if dropout:
      x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', name='fc2')(x)
    if dropout:
      x = Dropout(0.5)(x)
elif test == 't4':
    x = Flatten(name='flat')(x)
    x = Dense(1024, activation='relu', name='fc1')(x)
    if dropout:
      x = Dropout(0.5)(x)
elif test == 't5':
    x = Flatten(name='flat')(x)
    x = Dense(512, activation='relu', name='fc1')(x)
    if dropout:
      x = Dropout(0.5)(x)
elif test == 't6':
    x = Flatten(name='flat')(x)
    x = Dense(265, activation='relu', name='fc1')(x)
    if dropout:
      x = Dropout(0.5)(x)
elif test == 'baseline':
    x = base_model.get_layer('fc2').output
    
x = Dense(8, activation='softmax',name='predictions')(x)

model = Model(input=base_model.input, output=x)
plot(model, to_file='modelVGG16b_'+test+'.png', show_shapes=True, show_layer_names=True)
for layer in base_model.layers:
    layer.trainable = False
    
    
model.compile(loss='categorical_crossentropy',optimizer='adadelta', metrics=['accuracy'])
for layer in model.layers:
    print layer.name, layer.trainable

#preprocessing_function=preprocess_input,
if dataug:
  datagen = ImageDataGenerator(featurewise_center=False,
      samplewise_center=False,
      featurewise_std_normalization=False,
      samplewise_std_normalization=False,
      preprocessing_function = preprocess_input,
      rotation_range=0.,
      width_shift_range=0.1,
      height_shift_range=0.1,
      shear_range=0.,
      zoom_range=0,
      channel_shift_range=0.,
      fill_mode='nearest',
      cval=0.,
      horizontal_flip=True,
      vertical_flip=False,
      rescale=None)
else:
  datagen = ImageDataGenerator(featurewise_center=False,
      samplewise_center=False,
      featurewise_std_normalization=False,
      samplewise_std_normalization=False,
      preprocessing_function = preprocess_input,
      rotation_range=0.,
      width_shift_range=0.,
      height_shift_range=0.,
      shear_range=0.,
      zoom_range=0,
      channel_shift_range=0.,
      fill_mode='nearest',
      cval=0.,
      horizontal_flip=False,
      vertical_flip=False,
      rescale=None)

train_generator = datagen.flow_from_directory(train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

test_generator = datagen.flow_from_directory(test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

validation_generator = datagen.flow_from_directory(val_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

history=model.fit_generator(train_generator,
        samples_per_epoch=count_samples(train_data_dir),
        nb_epoch=number_of_epoch,
        validation_data=validation_generator,
        nb_val_samples=count_samples(val_data_dir))


result = model.evaluate_generator(test_generator, val_samples=count_samples(test_data_dir))

# Print the output of the evaluate generator
for i in range(len(result)):
  print model.metrics_names[i], result[i]
#print result

model.save_weights('baseline_weights.dat')

# list all data in history

if False:
  # summarize history for accuracy
  plt.plot(history.history['acc'])
  plt.plot(history.history['val_acc'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'validation'], loc='upper left')
  plt.savefig('accuracy.jpg')
  plt.close()
  # summarize history for loss
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'validation'], loc='upper left')
  plt.savefig('loss.jpg')
