import cv2
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn import preprocessing
import argparse
import os.path
# custom functions
from codebooks import compute_codebook
from utils import get_cross_val_dataset, normalize_vector
from features import features_detector
from svm_kernels import histogramIntersectionKernel


def getDescriptors(SIFTdetector, folds_data, pyramid):
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

			# depending on the parameter pyramid, the variables returned are different
			# for pyramid=False, kpt and des are lists of keypoints and descriptors (1D)
			# for pyramid=True, kpt and des are a list of 4 lists of keypoints and descriptors (2D)
			kpt, des = SIFTdetector.detect_compute(gray, pyramid)

			folds_descriptors[index]['descriptors'].append(des)
			folds_descriptors[index]['label_per_descriptor'].append(fold[1][i])
			if pyramid:
				print str(kpt.shape[0]*kpt.shape[1])+' extracted keypoints and descriptors'
			else:
				print str(len(kpt))+' extracted keypoints and descriptors'

	return folds_descriptors

class siftDetector(object):
	def getInitData(self, descriptor):
		size_descriptors=descriptor[0].shape[1]
		D=np.zeros((np.sum([len(p) for p in descriptor]),size_descriptors),dtype=np.uint8)

		return size_descriptors, D

	def getVisualWords(self, codebook, descriptor, size_descriptors, code_size):
		visual_words=np.zeros((len(descriptor),code_size),dtype=np.float32)

		for i in xrange(len(descriptor)):
			words=codebook.predict(descriptor[i].reshape(-1, size_descriptors))
			visual_words[i,:]=np.bincount(words,minlength=code_size)

		return visual_words

class pyramidDetector(object):
	def getInitData(self, descriptor):
		size_descriptors=descriptor[0].shape[2]
		D=np.zeros((np.sum([len(p.reshape(-1, size_descriptors)) for p in descriptor]),size_descriptors),dtype=np.uint8)

		return size_descriptors, D

	def getVisualWords(self, codebook, descriptor, size_descriptors, code_size):
		visual_words=np.zeros((len(descriptor),code_size*5),dtype=np.float32)

		for i in xrange(len(descriptor)):
			words=codebook.predict(descriptor[i].reshape(-1, size_descriptors))
			visual_words[i,:code_size] = np.bincount(words,minlength=code_size)
			visual_words[i,:code_size] = normalize_vector(visual_words[i,:code_size]) # normalize

			for j in range(4):
				words=codebook.predict(descriptor[i][j])
				visual_words[i,code_size*(j+1):code_size*(j+2)] = np.bincount(words,minlength=code_size)
				visual_words[i,code_size*(j+1):code_size*(j+2)] = normalize_vector(visual_words[i,code_size*(j+1):code_size*(j+2)]) # normalize

		return visual_words
	
def getDetector(pyramid):
	if pyramid:
		return pyramidDetector()
	else:
		return siftDetector()


def getCrossVal(folds_num, folds_descriptors, start, nfeatures, code_size, kernel, C, features, pyramid):
	accuracies = []
	for fold_i in range(folds_num): # 5 folds
		# Transform everything to numpy arrays
		Train_descriptors = []
		train_labels = []

		# select training images
		for j in range(folds_num):
			if fold_i != j:
				Train_descriptors.extend(folds_descriptors[j]['descriptors'])
				train_labels.extend(folds_descriptors[j]['label_per_descriptor'])

		detector = getDetector(pyramid)
		size_descriptors, D = detector.getInitData(Train_descriptors)

		#D=np.zeros((np.sum([len(p) for p in Train_descriptors]),size_descriptors),dtype=np.uint8)
		startingpoint=0
		for i in range(len(Train_descriptors)):
			descriptor_i = Train_descriptors[i].reshape(-1, size_descriptors)
			D[startingpoint:startingpoint+len(descriptor_i)]=descriptor_i
			startingpoint+=len(descriptor_i)

		# Compute Codebook
		codebook = compute_codebook(D, code_size, nfeatures, fold_i, features)

		# get visual words from BoW model
		init=time.time()

		visual_words = detector.getVisualWords(codebook, Train_descriptors, size_descriptors, code_size)

		end=time.time()
		print 'Done in '+str(end-init)+' secs.'

		# Train a linear SVM classifier
		stdSlr = StandardScaler().fit(visual_words)
		D_scaled = stdSlr.transform(visual_words)
		print 'Training the SVM classifier...'
		if kernel == 'intersection':
			ker_matrix = histogramIntersectionKernel(D_scaled, D_scaled)
			clf = svm.SVC(kernel='precomputed', C=C)
			clf.fit(ker_matrix, train_labels)
		else:
			clf = svm.SVC(kernel=kernel, C=C).fit(D_scaled, train_labels)

		print 'Done!'

		# get all the test data and predict their labels
		test_images_desc = folds_descriptors[fold_i]['descriptors']
		test_labels = folds_descriptors[fold_i]['label_per_descriptor']
		#visual_words_test=np.zeros((len(test_images_desc),code_size),dtype=np.float32)

		visual_words_test = detector.getVisualWords(codebook, test_images_desc, size_descriptors, code_size)

		if kernel == 'intersection':
			predictMatrix = histogramIntersectionKernel(stdSlr.transform(visual_words_test), D_scaled)
			accuracy = 100*clf.score(predictMatrix, test_labels)
		else:
			accuracy = 100*clf.score(stdSlr.transform(visual_words_test), test_labels)

		print 'Fold '+str(fold_i)+' accuracy: ' + str(accuracy)

		end=time.time()
		print 'Done in '+str(end-start)+' secs.'

		accuracies.append(accuracy)


		# show histogram instersection kernel
		#label_encoder_train = preprocessing.LabelEncoder()
		#label_encoder_train.fit(train_labels)

		#label_encoder_test = preprocessing.LabelEncoder()
		#label_encoder_test.fit(test_labels)

		#histogram_intersection_kernel(label_encoder_train.transform(train_labels), label_encoder_test.transform(test_labels), C, train_labels)
		#prediction = histogram_intersection_kernel(label_encoder_train.transform(train_labels), label_encoder_test.transform(test_labels), C, train_labels)

		## 49.56% in 285 secs.

	accuracies = np.asarray(accuracies)
	return accuracies


def main(nfeatures=100, code_size=512, n_components=60, kernel='linear', C=1, reduction=None, features='sift', pyramid=False):
	start = time.time()

	# read the train and test files
	folds_data = get_cross_val_dataset()

	# create the SIFT detector object
	SIFTdetector = features_detector(nfeatures, features)

	# read all the images per train
	# extract SIFT keypoints and descriptors
	# store descriptors in a python list of numpy arrays
	folds_descriptors = getDescriptors(SIFTdetector, folds_data, pyramid)
	
	# now perform de cross-val
	folds_num = 5
	accuracies = getCrossVal(folds_num, folds_descriptors, start, nfeatures, code_size,  kernel, C,features, pyramid)

	print('Final accuracy: %f (%f)' % (np.mean(accuracies), np.std(accuracies)))

parser = argparse.ArgumentParser()
parser.add_argument('-n_feat', help='Number of features per image to use', type=int, default=100)
parser.add_argument('-code_size', help='Codebook size', type=int, default=512)
parser.add_argument('-n_comp', help='Number of features to keep after feature reduction', type=int, default=60)
parser.add_argument('-kern', help='SVM kernel to use', type=str, default='intersection')
parser.add_argument('-C', help='SVM C parameter', type=float, default=1.0)
parser.add_argument('-reduce', help='Feature reduction', type=str, default=None)
parser.add_argument('-feats', help='Features to use', type=str, default='sift')
parser.add_argument('--pyramid', dest='pyramid', action='store_true')
args = parser.parse_args()

print(args)

main(args.n_feat, args.code_size, args.n_comp, args.kern, args.C, args.reduce, args.feats, args.pyramid)
