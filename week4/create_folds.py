from utils import get_dataset
import numpy as np
import os
import cPickle

train_images_filenames, _, train_labels, _ = get_dataset(verbose=False)
train_images_filenames = np.asarray(train_images_filenames)
train_labels = np.asarray(train_labels)

# shuffle lists
assert len(train_images_filenames) == len(train_labels)
permutation = np.random.permutation(len(train_images_filenames))
train_images_filenames = train_images_filenames[permutation]
train_labels = train_labels[permutation]

# split in 5 folds
images_folds = np.array_split(train_images_filenames, 5)
labels_folds = np.array_split(train_labels, 5)

if not os.path.exists('folds'):
	os.mkdir('folds')

for i,fold in enumerate(zip(images_folds, labels_folds)):
	print(i)
	cPickle.dump(fold[0], open('folds/images_filenames_fold_'+str(i)+'.dat', "wb"))
	cPickle.dump(fold[1], open('folds/labels_fold_'+str(i)+'.dat', "wb"))
