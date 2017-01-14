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

			print str(len(kpt)*len(kpt[0]))+' extracted keypoints and descriptors'

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
		print str(len(kpt)*len(kpt[0]))+' extracted keypoints and descriptors'

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

		# if pyramid:

		# 	# level 0
		# 	keypoints = [kpt]
		# 	descriptors = [des]

		# 	config = [(2,2),(3,3)]

		# 	for level in range()

		# 	print(gray.shape)
		# 	print(len(kpt))
		# 	print(kpt[0].pt)
		# 	print(kpt[0].octave)
		# 	print(kpt[-1].pt)
		# 	print(kpt[-1].octave)

		# 	quit()

		# 	kpt = []
		# 	des = []
		# 	middle_x = int(gray.shape[0]/2.0)
		# 	middle_y = int(gray.shape[1]/2.0)
			
		# 	# extract keyoints and descriptors by level in a 2x2 grid
		# 	#########
		# 	# 1 | 2 #
		# 	# ----- #
		# 	# 3 | 4 #
		# 	#########

		# 	level1_kpt, level1_des = self.get_descriptors(gray[:middle_x, :middle_y])
		# 	level2_kpt, level2_des = self.get_descriptors(gray[middle_x:, :middle_y])
		# 	level3_kpt, level3_des = self.get_descriptors(gray[:middle_x, middle_y:])
		# 	level4_kpt, level4_des = self.get_descriptors(gray[middle_x:, middle_y:])

		# 	kpt = np.asarray([level1_kpt, level2_kpt, level3_kpt, level4_kpt])
		# 	des = np.asarray([level1_des, level2_des, level3_des, level4_des])
		# else:
		# 	kpt, des = self.get_descriptors(gray)

		return keypoints, descriptors
		
		
def features_detector(nfeatures=100, features='sift', grid_step=6):
	detector = CustomDetector(features, nfeatures, grid_step)
	detector.set_detector()
	detector.set_extractor()

	return detector