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

def main(nfeatures=100, code_size=32, n_components=60, kernel='linear', C=1, reduction=None, features='sift', pyramid=False):
	start = time.time()

	# read the train and test files
	train_images_filenames, test_images_filenames, train_labels, test_labels = get_dataset()

	# create the SIFT detector object

	SIFTdetector = cv2.SIFT(nfeatures=nfeatures)

	# extract SIFT keypoints and descriptors
	# store descriptors in a python list of numpy arrays

	Train_descriptors = []
	Train_label_per_descriptor = []

	for i in range(len(train_images_filenames)):
		filename=train_images_filenames[i]
		print 'Reading image '+filename
		ima=cv2.imread(filename)
		gray=cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
		kpt,des=SIFTdetector.detectAndCompute(gray,None)
		Train_descriptors.append(des)
		Train_label_per_descriptor.append(train_labels[i])
		print str(len(kpt))+' extracted keypoints and descriptors'

	# Transform everything to numpy arrays
	size_descriptors=Train_descriptors[0].shape[1]
	D=np.zeros((np.sum([len(p) for p in Train_descriptors]),size_descriptors),dtype=np.uint8)
	startingpoint=0
	for i in range(len(Train_descriptors)):
		D[startingpoint:startingpoint+len(Train_descriptors[i])]=Train_descriptors[i]
		startingpoint+=len(Train_descriptors[i])


	k = code_size

	print 'Computing gmm with '+str(k)+' centroids'
	init=time.time()
	gmm = ynumpy.gmm_learn(np.float32(D), k)
	end=time.time()
	print 'Done in '+str(end-init)+' secs.'



	init=time.time()
	fisher=np.zeros((len(Train_descriptors),k*128*2),dtype=np.float32)
	for i in xrange(len(Train_descriptors)):
		fisher[i,:]= ynumpy.fisher(gmm, Train_descriptors[i], include = ['mu','sigma'])


	end=time.time()
	print 'Done in '+str(end-init)+' secs.'


	# Train a linear SVM classifier

	stdSlr = StandardScaler().fit(fisher)
	D_scaled = stdSlr.transform(fisher)
	print 'Training the SVM classifier...'
	clf = svm.SVC(kernel=kernel, C=C).fit(D_scaled, train_labels)
	print 'Done!'

	# get all the test data and predict their labels
	fisher_test=np.zeros((len(test_images_filenames),k*128*2),dtype=np.float32)
	for i in range(len(test_images_filenames)):
		filename=test_images_filenames[i]
		print 'Reading image '+filename
		ima=cv2.imread(filename)
		gray=cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
		kpt,des=SIFTdetector.detectAndCompute(gray,None)
		fisher_test[i,:]=ynumpy.fisher(gmm, des, include = ['mu','sigma'])


	accuracy = 100*clf.score(stdSlr.transform(fisher_test), test_labels)

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
parser.add_argument('-feats', help='Features to use', type=str, default='sift')
parser.add_argument('--pyramid', dest='pyramid', action='store_true')
args = parser.parse_args()

print(args)

main(args.n_feat, args.code_size, args.n_comp, args.kern, args.C, args.reduce, args.feats, args.pyramid)