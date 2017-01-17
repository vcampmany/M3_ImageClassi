import cv2
import numpy as np
import cPickle
import time
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn import cluster
from yael import ynumpy
from utils import get_dataset
import argparse
from data import getDescriptors, features_detector, PCA_reduce
from codebooks import compute_codebook

def main(nfeatures=100, code_size=32, n_components=60, kernel='linear', C=1, reduction=None, features='sift', pyramid=False, grid_step=6):
	start = time.time()

	# read the train and test files
	train_images_filenames, test_images_filenames, train_labels, test_labels = get_dataset()

	# create the SIFT detector object
	SIFTdetector = features_detector(nfeatures, features, grid_step)

	# extract SIFT keypoints and descriptors
	# store descriptors in a python list of numpy arrays

	Train_descriptors, Train_label_per_descriptor = getDescriptors(SIFTdetector, train_images_filenames, train_labels, pyramid)

	# Transform everything to numpy arrays
	size_descriptors=Train_descriptors[0].shape[1]
	D=np.zeros((np.sum([len(p) for p in Train_descriptors]),size_descriptors),dtype=np.uint8)
	startingpoint=0
	for i in range(len(Train_descriptors)):
		D[startingpoint:startingpoint+len(Train_descriptors[i])]=Train_descriptors[i]
		startingpoint+=len(Train_descriptors[i])
	if reduction == 'pca':
		D, pca_reducer = PCA_reduce(D, n_components)

	k = code_size
	# Compute Codebook
	gmm = compute_codebook(D, k, nfeatures, None, features, D.shape[1])

	init=time.time()
	fisher=np.zeros((len(Train_descriptors),k*D.shape[1]*2),dtype=np.float32)
	for i in xrange(len(Train_descriptors)):
		fisher[i,:]= ynumpy.fisher(gmm, Train_descriptors[i], include = ['mu','sigma'])

	end=time.time()
	print 'Done in '+str(end-init)+' secs.'

	# Train a linear SVM classifier

	stdSlr = StandardScaler().fit(fisher)
	D_scaled = stdSlr.transform(fisher)
	print 'Training the SVM classifier...'
	
	if kernel == 'pyramid_match':
		ker_matrix = spatialPyramidKernel(D_scaled, D_scaled, k*D.shape[1]*2, pyramid)
		clf = svm.SVC(kernel='precomputed', C=C)
		clf.fit(ker_matrix, train_labels)
	else:
		clf = svm.SVC(kernel=kernel, C=C).fit(D_scaled, train_labels)
	print 'Done!'

	# get all the test data and predict their labels
	fisher_test=np.zeros((len(test_images_filenames),k*D.shape[1]*2),dtype=np.float32)
	for i in range(len(test_images_filenames)):
		filename=test_images_filenames[i]
		print 'Reading image '+filename
		ima=cv2.imread(filename)
		gray=cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
		kpt,des=SIFTdetector.detect_compute(gray)
		if reduction == 'pca':
			des = pca_reducer.transform(des)
		fisher_test[i,:]=ynumpy.fisher(gmm, des, include = ['mu','sigma'])

	
	accuracy = 100*clf.score(stdSlr.transform(fisher_test), test_labels)
	fisher_test = stdSlr.transform(fisher_test)
	if kernel == 'pyramid_match':
		predictMatrix = spatialPyramidKernel(fisher_test, D_scaled, k*D.shape[1]*2, pyramid)
		#predictions = clf.predict(predictMatrix)
		#predictions_proba = clf.predict_proba(predictMatrix)
		accuracy = 100*clf.score(predictMatrix, test_labels)
	else:
		accuracy = 100*clf.score(fisher_test, test_labels)

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
parser.add_argument('-feats', help='Features to use', type=str, default='dense_sift')
parser.add_argument('-grid_step', help='step of the sift grid', type=int, default=6)
parser.add_argument('--pyramid', dest='pyramid', action='store_true')
args = parser.parse_args()

print(args)

main(args.n_feat, args.code_size, args.n_comp, args.kern, args.C, args.reduce, args.feats, args.pyramid, args.grid_step)