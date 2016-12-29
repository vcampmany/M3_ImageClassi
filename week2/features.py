import cv2

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

	def set_extractor(self):
		if self.features == 'sift':
			self.extractor = self.detector
		elif self.features == 'dense_sift':
			self.extractor = cv2.SIFT()

	def detect_compute(self, gray):
		if self.features == 'sift':
			kpt = self.detector.detect(gray,None)
			kpt,des=self.extractor.compute(gray,kpt)
		elif self.features == 'dense_sift':
			kpt=self.detector.detect(gray)
			kpt,des=self.extractor.compute(gray,kpt)

		return kpt, des
		
def features_detector(nfeatures=100, features='sift'):
	detector = CustomDetector(features, nfeatures)
	detector.set_detector()
	detector.set_extractor()

	return detector