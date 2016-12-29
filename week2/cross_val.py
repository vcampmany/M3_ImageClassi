import cv2
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import argparse
import os.path
# custom functions
from codebooks import compute_codebook
from utils import get_cross_val_dataset
from features import features_detector

def main(nfeatures=100, code_size=512, n_components=60, kernel='linear', C=1, reduction=None, features='sift'):
	start = time.time()

	# read the train and test files
	folds_data = get_cross_val_dataset()

	# create the SIFT detector object
	SIFTdetector = features_detector(nfeatures, features)

	# read all the images per train
	# extract SIFT keypoints and descriptors
	# store descriptors in a python list of numpy arrays

	folds_descriptors = {}

	for index,fold in enumerate(folds_data):
		folds_descriptors[index] = {
			'descriptors': [],
			'label_per_descriptor': []
		}
		for i in xrange(len(fold[0])):
			filename=fold[0][i]
			print 'Reading image '+filename
			ima=cv2.imread(filename)
			gray=cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
			# kpt = SIFTdetector.detect(gray,None)
			# kpt,des=SIFTdetector.compute(gray,kpt)

			kpt, des = SIFTdetector.detect_compute(gray)

			folds_descriptors[index]['descriptors'].append(des)
			folds_descriptors[index]['label_per_descriptor'].append(fold[1][i])
			print str(len(kpt))+' extracted keypoints and descriptors'

	accuracies = []
	# now perform de cross-val 
	for fold_i in range(5): # 5 folds
		# Transform everything to numpy arrays
		Train_descriptors = []
		train_labels = []

		# select training images
		for j in range(5):
			if fold_i != j:
				Train_descriptors.extend(folds_descriptors[j]['descriptors'])
				train_labels.extend(folds_descriptors[j]['label_per_descriptor'])

		size_descriptors=Train_descriptors[0].shape[1]
		D=np.zeros((np.sum([len(p) for p in Train_descriptors]),size_descriptors),dtype=np.uint8)
		startingpoint=0
		for i in range(len(Train_descriptors)):
			D[startingpoint:startingpoint+len(Train_descriptors[i])]=Train_descriptors[i]
			startingpoint+=len(Train_descriptors[i])

		# Compute Codebook
		codebook = compute_codebook(D, code_size, nfeatures, fold_i, features)

		# get visual words from BoW model
		init=time.time()
		visual_words=np.zeros((len(Train_descriptors),code_size),dtype=np.float32)
		for i in xrange(len(Train_descriptors)):
			words=codebook.predict(Train_descriptors[i])
			visual_words[i,:]=np.bincount(words,minlength=code_size)

		end=time.time()
		print 'Done in '+str(end-init)+' secs.'

		# Train a linear SVM classifier

		stdSlr = StandardScaler().fit(visual_words)
		D_scaled = stdSlr.transform(visual_words)
		print 'Training the SVM classifier...'
		clf = svm.SVC(kernel=kernel, C=C).fit(D_scaled, train_labels)
		print 'Done!'

		# get all the test data and predict their labels
		test_images_desc = folds_descriptors[fold_i]['descriptors']
		test_labels = folds_descriptors[fold_i]['label_per_descriptor']
		visual_words_test=np.zeros((len(test_images_desc),code_size),dtype=np.float32)
		for i in range(len(test_images_desc)):
			# filename=test_images_filenames[i]
			# print 'Reading image '+filename
			# ima=cv2.imread(filename)
			# gray=cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
			# kpt,des=SIFTdetector.detectAndCompute(gray,None)
			words=codebook.predict(test_images_desc[i])
			visual_words_test[i,:]=np.bincount(words,minlength=code_size)


		accuracy = 100*clf.score(stdSlr.transform(visual_words_test), test_labels)

		print 'Fold '+str(fold_i)+' accuracy: ' + str(accuracy)

		end=time.time()
		print 'Done in '+str(end-start)+' secs.'

		accuracies.append(accuracy)

		## 49.56% in 285 secs.

	accuracies = np.asarray(accuracies)
	print('Final accuracy: %f (%f)' % (np.mean(accuracies), np.std(accuracies)))

parser = argparse.ArgumentParser()
parser.add_argument('-n_feat', help='Number of features per image to use', type=int, default=100)
parser.add_argument('-code_size', help='Codebook size', type=int, default=512)
parser.add_argument('-n_comp', help='Number of features to keep after feature reduction', type=int, default=60)
parser.add_argument('-kern', help='SVM kernel to use', type=str, default='linear')
parser.add_argument('-C', help='SVM C parameter', type=float, default=1.0)
parser.add_argument('-reduce', help='Feature reduction', type=str, default=None)
parser.add_argument('-feats', help='Features to use', type=str, default='sift')
args = parser.parse_args()

print(args)

main(args.n_feat, args.code_size, args.n_comp, args.kern, args.C,  args.reduce, args.feats)