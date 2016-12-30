import cPickle

def get_dataset(verbose=True):
	train_images_filenames = cPickle.load(open('../train_images_filenames.dat','r'))
	train_images_filenames = [filename.replace('../../Databases/', '../') for filename in train_images_filenames]
	test_images_filenames = cPickle.load(open('../test_images_filenames.dat','r'))
	test_images_filenames = [filename.replace('../../Databases/', '../') for filename in test_images_filenames]
	train_labels = cPickle.load(open('../train_labels.dat','r'))
	test_labels = cPickle.load(open('../test_labels.dat','r'))

	if verbose:
		print 'Loaded '+str(len(train_images_filenames))+' training images filenames with classes ',set(train_labels)
		print 'Loaded '+str(len(test_images_filenames))+' testing images filenames with classes ',set(test_labels)

	return train_images_filenames, test_images_filenames, train_labels, test_labels

def get_cross_val_dataset(verbose=True, folds=5):
	folds_data = []
	for fold in range(folds):
		images_filenames = cPickle.load(open('folds/images_filenames_fold_'+str(fold)+'.dat','r'))
		labels = cPickle.load(open('folds/labels_fold_'+str(fold)+'.dat','r'))

		if verbose:
			print 'Loaded '+str(len(images_filenames))+' images filenames with classes ',set(labels)

		folds_data.append((images_filenames, labels))

	return folds_data

def normalize_vector(vec):
	normed = vec / vec.sum()

	return normed