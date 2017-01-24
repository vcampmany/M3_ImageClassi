import numpy as np
import cPickle
import time
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn import cluster
from yael import ynumpy
from utils import get_dataset
import argparse
from data import getDescriptors, cnn_features, PCA_reduce, subsample_feature
from codebooks import compute_codebook

def main(nfeatures=100, code_size=512, n_components=60, kernel='linear', C=1, reduction=None, output_layer='fc2', decision='svm', sampling_step=4, sampling_type='default'):
	start = time.time()

	# read the train and test files
	train_images_filenames, test_images_filenames, train_labels, test_labels = get_dataset()

	# create the CNN detector object
	cnn_model = cnn_features(output_layer)

	# extract SIFT keypoints and descriptors
	# store descriptors in a python list of numpy arrays

	Train_descriptors, Train_label_per_descriptor = getDescriptors(cnn_model, train_images_filenames, train_labels, decision, sampling_step, sampling_type)

	Train_descriptors = np.asarray(Train_descriptors)

	# Transform everything to numpy arrays
	size_descriptors=Train_descriptors[0][0].shape[-1]
	# for D we only need the first level of the pyramid (because it already contains all points)
	D=np.zeros((np.sum([len(p[0]) for p in Train_descriptors]),size_descriptors),dtype=np.uint8)
	startingpoint=0
	for i in range(len(Train_descriptors)):
		D[startingpoint:startingpoint+len(Train_descriptors[i][0])]=Train_descriptors[i][0]
		startingpoint+=len(Train_descriptors[i][0])
	if reduction == 'pca':
		D, pca_reducer = PCA_reduce(D, n_components)

	if decision == 'bow':
		k = code_size
		# Compute Codebook
		gmm = compute_codebook(D, k, nfeatures, None, output_layer, D.shape[1], sampling_step, sampling_type)

		init=time.time()
		samples=np.zeros((len(Train_descriptors),k*D.shape[1]*2*Train_descriptors.shape[1]),dtype=np.float32)  #TODO: change 128
		for i in xrange(len(Train_descriptors)):
			for j in range(Train_descriptors.shape[1]): #number of levels
				if reduction == 'pca':
					des = pca_reducer.transform(Train_descriptors[i][j]) # for pyramid level j
				else:
					des = Train_descriptors[i][j] # for pyramid level j
				samples[i,j*k*D.shape[1]*2:(j+1)*k*D.shape[1]*2]= ynumpy.fisher(gmm, np.float32(des), include = ['mu','sigma'])

		end=time.time()
		print 'Done in '+str(end-init)+' secs.'
	else:
		samples = D

	# Train a linear SVM classifier
	stdSlr = StandardScaler().fit(samples)
	D_scaled = stdSlr.transform(samples)

	print 'Training the SVM classifier...'
	clf = svm.SVC(kernel=kernel, C=C).fit(D_scaled, train_labels)
	print 'Done!'


	# Apply BoW
	if decision == 'bow':

		test_descriptors, test_label_per_descriptor = getDescriptors(cnn_model, test_images_filenames, test_labels, decision, sampling_step, sampling_type)
		test_descriptors = np.asarray(test_descriptors)

		fisher_test=np.zeros((len(test_descriptors),k*D.shape[1]*2*Train_descriptors.shape[1]),dtype=np.float32)

		for i in range(len(test_descriptors)):
			for j in range(test_descriptors.shape[1]): #number of levels
				des = test_descriptors[i][j] # now only working with 1 PYRAMID LEVEL [0]
				if reduction == 'pca':
					des = pca_reducer.transform(des)
				fisher_test[i,j*k*D.shape[1]*2:(j+1)*k*D.shape[1]*2]=ynumpy.fisher(gmm, np.float32(des), include = ['mu','sigma'])
		test_images_desc = fisher_test
	else:
		test_descriptors, test_label_per_descriptor = getDescriptors(cnn_model, test_images_filenames, test_labels, decision, sampling_step, sampling_type)
		test_descriptors = np.asarray(test_descriptors)

		test_images_desc = test_descriptors.squeeze()
		if reduction == 'pca':
			test_images_desc = pca_reducer.transform(test_images_desc)

	test_images_desc = stdSlr.transform(test_images_desc)
	accuracy = 100*clf.score(test_images_desc, test_labels)	
	
	print 'Final accuracy: ' + str(accuracy)

	end=time.time()
	print 'Done in '+str(end-start)+' secs.'

	## 61.71% in 251 secs.

parser = argparse.ArgumentParser()
parser.add_argument('-n_feat', help='Number of features per image to use', type=int, default=100)
parser.add_argument('-code_size', help='Codebook size', type=int, default=32)
parser.add_argument('-n_comp', help='Number of features to keep after feature reduction', type=int, default=60)
parser.add_argument('-kern', help='SVM kernel to use', type=str, default='linear')
parser.add_argument('-C', help='SVM C parameter', type=float, default=1.0)
parser.add_argument('-reduce', help='Feature reduction', type=str, default=None)
parser.add_argument('-output_layer', help='output layer', type=str, default='fc2')
parser.add_argument('-decision', help='svm or bow ', type=str, default='svm')
parser.add_argument('-sampling_step', help='step of the subsampling', type=int, default=4)
parser.add_argument('-sampling_type', help='Type of the subsampling ("default" or "average")', type=str, default='default')
args = parser.parse_args()

print(args)

main(args.n_feat, args.code_size, args.n_comp, args.kern, args.C, args.reduce, args.output_layer, args.decision, args.sampling_step, args.sampling_type)