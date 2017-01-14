'''
### This file is for functions related to dealing with data:
###		- reading files, extracting features...
'''

import cv2
import numpy as np
from sklearn.decomposition import PCA
import pyramids

def PCA_reduce(D, n_components):
	print(D.shape)
	pca = PCA(n_components=n_components)
	pca.fit(D)
	return pca.transform(D), pca

def getFoldsDescriptors(SIFTdetector, folds_data, pyramid):
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

			# the dimensions are: LxNx128 where N: number of points, L=number of pyramid levels
			kpt, des = SIFTdetector.detect_compute(gray, pyramid)

			folds_descriptors[index]['descriptors'].append(des)
			folds_descriptors[index]['label_per_descriptor'].append(fold[1][i])

			print str(sum([len(level) for level in kpt]))+' extracted keypoints and descriptors'

	return folds_descriptors

def getDescriptors(SIFTdetector, images_filenames, labels, pyramid):
	descriptors = []
	label_per_descriptor = []

	for i in xrange(len(images_filenames)):
		filename=images_filenames[i]
		print 'Reading image '+filename
		ima=cv2.imread(filename)
		gray=cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)

		# the dimensions are: LxNx128 where N: number of points, L=number of pyramid levels
		kpt, des = SIFTdetector.detect_compute(gray, pyramid)

		descriptors.append(des)
		label_per_descriptor.append(labels[i])
		print str(sum([len(level) for level in kpt]))+' extracted keypoints and descriptors'

	return descriptors, label_per_descriptor

class CustomDetector(object):
	"""docstring for CustomDetector"""
	def __init__(self, features, nfeatures, grid_step):
		super(CustomDetector, self).__init__()
		self.features = features
		self.nfeatures = nfeatures
		self.detector = None
		self.extractor = None
		self.grid_step = grid_step

	def set_detector(self):
		if self.features == 'sift':
			self.detector = cv2.SIFT(nfeatures=self.nfeatures)
		elif self.features == 'dense_sift':
			self.detector = cv2.FeatureDetector_create("Dense")
			# dense.setDouble('initFeatureScale', 10)
			self.detector.setInt('initXyStep', self.grid_step)

	def set_extractor(self):
		if self.features == 'sift':
			self.extractor = self.detector
		elif self.features == 'dense_sift':
			self.extractor = cv2.SIFT()

	def get_descriptors(self, gray):
		if self.features == 'sift':
			kpt = self.detector.detect(gray,None)
			kpt,des=self.extractor.compute(gray,kpt)
		elif self.features == 'dense_sift':
			kpt=self.detector.detect(gray)
			kpt,des=self.extractor.compute(gray,kpt)

		return kpt, des

	def detect_compute(self, gray, pyramid=None):

		kpt, des = self.get_descriptors(gray)

		# level 0
		keypoints = [kpt]
		descriptors = [des]

		if pyramid: # extract more levels
			levels = pyramids.get_levels(pyramid)

			pyramid_kps, pyramid_des = pyramids.extract_pyramid_bins(levels, kpt, des, [0,0,gray.shape[0],gray.shape[1]])
			keypoints += pyramid_kps
			descriptors += pyramid_des

		return keypoints, descriptors
		
		
def features_detector(nfeatures=100, features='sift', grid_step=6):
	detector = CustomDetector(features, nfeatures, grid_step)
	detector.set_detector()
	detector.set_extractor()

	return detector