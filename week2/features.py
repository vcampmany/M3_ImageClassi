import cv2
import numpy as np

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