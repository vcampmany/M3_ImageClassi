'''
### This file is for functions related to dealing with data:
###		- reading files, extracting features...
'''

import cv2
import numpy as np

def getDescriptors(SIFTdetector, images_filenames, labels, pyramid):

	descriptors = []
	label_per_descriptor = []

	for i in xrange(len(images_filenames)):
		filename=images_filenames[i]
		print 'Reading image '+filename
		ima=cv2.imread(filename)
		gray=cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)

		# depending on the parameter pyramid, the variables returned are different
		# for pyramid=False, kpt and des are lists of keypoints and descriptors (1D)
		# for pyramid=True, kpt and des are a list of 4 lists of keypoints and descriptors (2D)
		kpt, des = SIFTdetector.detect_compute(gray, pyramid)

		descriptors.append(des)
		label_per_descriptor.append(labels[i])
		if pyramid:
			print str(kpt.shape[0]*kpt.shape[1])+' extracted keypoints and descriptors'
		else:
			print str(len(kpt))+' extracted keypoints and descriptors'

	return descriptors, label_per_descriptor

class CustomDetector(object):
	"""docstring for CustomDetector"""
	def __init__(self, features, nfeatures):
		super(CustomDetector, self).__init__()
		self.features = features
		self.nfeatures = nfeatures
		self.detector = None
		self.extractor = None

	def set_detector(self):
		if self.features == 'sift':
			self.detector = cv2.SIFT(nfeatures=self.nfeatures)
		elif self.features == 'dense_sift':
			self.detector = cv2.FeatureDetector_create("Dense")
			# dense.setDouble('initFeatureScale', 10)
			# dense.setInt('initXyStep', 3)

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
		if pyramid:
			kpt = []
			des = []
			middle_x = int(gray.shape[0]/2.0)
			middle_y = int(gray.shape[1]/2.0)
			
			# extract keyoints and descriptors by level in a 2x2 grid
			#########
			# 1 | 2 #
			# ----- #
			# 3 | 4 #
			#########

			level1_kpt, level1_des = self.get_descriptors(gray[:middle_x, :middle_y])
			level2_kpt, level2_des = self.get_descriptors(gray[middle_x:, :middle_y])
			level3_kpt, level3_des = self.get_descriptors(gray[:middle_x, middle_y:])
			level4_kpt, level4_des = self.get_descriptors(gray[middle_x:, middle_y:])

			kpt = np.asarray([level1_kpt, level2_kpt, level3_kpt, level4_kpt])
			des = np.asarray([level1_des, level2_des, level3_des, level4_des])
		else:
			kpt, des = self.get_descriptors(gray)
		return kpt, des
		
		
def features_detector(nfeatures=100, features='sift'):
	detector = CustomDetector(features, nfeatures)
	detector.set_detector()
	detector.set_extractor()

	return detector