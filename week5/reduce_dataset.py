from os import listdir
from os.path import isfile, join
from PIL import Image
import os
import random

def mkdirs(directory):
	if not os.path.exists(directory):
		os.makedirs(directory)

samples = 8
n_classes = 8
samp_per_class = samples // n_classes

BASE_PATH = '/home/master/M3_ImageClassi/MIT_split/train/'
SAVE_PATH = '/home/master/M3_ImageClassi/MIT_split/train_'+str(samples)+'/'

classes = ['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding']

for label in classes:
	label_path = BASE_PATH+label
	onlyfiles = [f for f in listdir(label_path) if isfile(join(label_path, f))]
	random.shuffle(onlyfiles)
	onlyfiles = onlyfiles[0:samp_per_class]
	assert len(onlyfiles) == samp_per_class

	# make sure that the target directory exists
	target_path = SAVE_PATH+label
	mkdirs(target_path)

	for i in range(len(onlyfiles)):
		im = Image.open(label_path+'/'+onlyfiles[i])
		im.save(target_path+'/'+onlyfiles[i])