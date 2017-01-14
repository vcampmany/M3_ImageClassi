'''
### This file is for functions related to dealing with data:
###		- reading files, extracting features...
'''

import cv2
import numpy as np
from sklearn.decomposition import PCA

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

	def detect_compute(self, gray, pyramid=False):

		kpt, des = self.get_descriptors(gray)

		# level 0
		keypoints = [kpt]
		descriptors = [des]

		if pyramid: # extract more levels
			levels = [(2,2)]

			for level in levels:
				x_divisions, y_divisions = level

				min_limit_x, min_limit_y = 0,0
				max_limit_x, max_limit_y = gray.shape

				nbins = x_divisions*y_divisions # number of spatial bins
				# initialize lists for this level
				kpt_level_bins = [[[] for yi in range(y_divisions)] for xi in range(x_divisions)]
				des_level_bins = [[[] for yi in range(y_divisions)] for xi in range(x_divisions)]

				x_step = max_limit_x / float(x_divisions)
				y_step = max_limit_y / float(y_divisions)

				for x_div in range(x_divisions):
					for y_div in range(y_divisions):
						for i, kp in enumerate(kpt):
							min_x,max_x = x_step*x_div, x_step*(x_div+1)
							min_y,max_y = y_step*y_div, y_step*(y_div+1)
							# check if this keypoint belongs to the current bin
							if (kp.pt[0] >= min_x and kp.pt[0] < max_x) and (kp.pt[1] >= min_y and kp.pt[1] < max_y):
								# it belongs!
								kpt_level_bins[x_div][y_div].append(kpt[i])
								des_level_bins[x_div][y_div].append(des[i])

				# now append the bin descriptors and keypoints
				for x_div in range(x_divisions):
					for y_div in range(y_divisions):
						keypoints.append(kpt_level_bins[x_div][y_div])
						descriptors.append(des_level_bins[x_div][y_div])

		return keypoints, descriptors
		
		
def features_detector(nfeatures=100, features='sift', grid_step=6):
	detector = CustomDetector(features, nfeatures, grid_step)
	detector.set_detector()
	detector.set_extractor()

	return detector