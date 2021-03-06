'''
### This file is for functions related to dealing with data:
###		- reading files, extracting features...
'''

import numpy as np
from sklearn.decomposition import PCA
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from keras import backend as K
from keras.utils.visualize_util import plot

def PCA_reduce(D, n_components):
	print(D.shape)
	pca = PCA(n_components=n_components)
	pca.fit(D)
	return pca.transform(D), pca

def subsample_feature(tensor, red_ratio, sampling_type):
	size = (tensor.shape[0]/red_ratio, tensor.shape[1]/red_ratio, tensor.shape[2])
	sub_sampled = np.zeros(size)

	for i in range(sub_sampled.shape[0]):
		for j in range(sub_sampled.shape[1]):
			row = i*red_ratio
			col = j*red_ratio
			if sampling_type == 'avg':
				sub_sampled[i][j][:] = tensor[row:row+red_ratio, col:col+red_ratio, :].mean(axis=0).mean(axis=0)
			elif sampling_type == 'default':
				sub_sampled[i][j][:] = tensor[row, col, :]
			else:
				print('Unknown value for sampling_type')
				quit()
	return sub_sampled


def getFoldsDescriptors(model, folds_data, decision, sampling_step, sampling_type):
	folds_descriptors = {}

	for index,fold in enumerate(folds_data):
		folds_descriptors[index] = {
			'descriptors': [],
			'label_per_descriptor': []
		}
		for i in xrange(len(fold[0])):
			filename=fold[0][i]
			print 'Reading image '+filename

			# Read the image and preprocess
			img = image.load_img(filename, target_size=(224, 224))
			x = image.img_to_array(img)
			x = np.expand_dims(x, axis=0)
			x = preprocess_input(x)

			# do the prediction
			features = [model.predict(x)]

			if 'bow' == decision or 'fisher' == decision:
				features = features[0]
				features = np.squeeze(features)
				features = subsample_feature(features, sampling_step, sampling_type)
				features = np.reshape(features, (features.shape[0]*features.shape[1], features.shape[2]))
				features = [features]
				
			folds_descriptors[index]['descriptors'].append(features)
			folds_descriptors[index]['label_per_descriptor'].append(fold[1][i])

	return folds_descriptors

def getDescriptors(model, images_filenames, labels, decision, sampling_step, sampling_type):
	descriptors = []
	label_per_descriptor = []

	for i in xrange(len(images_filenames)):
		filename=images_filenames[i]
		print 'Reading image '+filename

		# Read the image and preprocess
		img = image.load_img(filename, target_size=(224, 224))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)

		# do the prediction
		features = [model.predict(x)]

		if 'bow' == decision or 'fisher' == decision:
			features = features[0]
			features = np.squeeze(features)
			features = subsample_feature(features, sampling_step, sampling_type)
			features = np.reshape(features, (features.shape[0]*features.shape[1], features.shape[2]))
			features = [features]

		descriptors.append(features)
		label_per_descriptor.append(labels[i])

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

def cnn_features(output_layer):
	#load VGG model
	base_model = VGG16(weights='imagenet')
	#crop the model up to a certain layer
	model = Model(input=base_model.input, output=base_model.get_layer(output_layer).output)
	return model
	