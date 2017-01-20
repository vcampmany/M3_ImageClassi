import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn import preprocessing

import argparse
from utils import get_cross_val_dataset, normalize_vector, l2_normalize_vector
from data import getFoldsDescriptors, cnn_features, PCA_reduce
from yael import ynumpy
from codebooks import compute_codebook

def getCrossVal(folds_num, folds_descriptors, start, nfeatures, code_size, kernel, C, features, pyramid, grid_step, n_comps, reduction):
	accuracies = []

	for fold_i in range(folds_num): # 5 folds
		# Transform everything to numpy arrays
		Train_descriptors = []
		train_labels = []

		# select training images
		for j in range(folds_num):
			if fold_i != j:
				Train_descriptors.extend(folds_descriptors[j]['descriptors'])
				train_labels.extend(folds_descriptors[j]['label_per_descriptor'])

		Train_descriptors = np.asarray(Train_descriptors)

		# print(Train_descriptors.shape)
		# print(Train_descriptors[0][0].shape)
		# quit()

		# Transform everything to numpy arrays
		size_descriptors=Train_descriptors[0][0].shape[-1]
		# for D we only need the first level of the pyramid (because it already contains all points)
		D=np.zeros((np.sum([len(p[0]) for p in Train_descriptors]),size_descriptors),dtype=np.uint8)
		startingpoint=0
		for i in range(len(Train_descriptors)):
			D[startingpoint:startingpoint+len(Train_descriptors[i][0])]=Train_descriptors[i][0]
			startingpoint+=len(Train_descriptors[i][0])
		if reduction == 'pca':
			D, pca_reducer = PCA_reduce(D, n_comps)

		k = code_size
		# Compute Codebook
		gmm = compute_codebook(D, k, nfeatures, fold_i, features, grid_step, D.shape[1])

		init=time.time()
		fisher=np.zeros((len(Train_descriptors),k*D.shape[1]*2*Train_descriptors.shape[1]),dtype=np.float32)  #TODO: change 128
		for i in xrange(len(Train_descriptors)):
			for j in range(Train_descriptors.shape[1]): #number of levels
				if reduction == 'pca':
					des = pca_reducer.transform(Train_descriptors[i][j]) # for pyramid level j
				else:
					des = Train_descriptors[i][j] # for pyramid level j
				fisher[i,j*k*D.shape[1]*2:(j+1)*k*D.shape[1]*2]= ynumpy.fisher(gmm, np.float32(des), include = ['mu','sigma'])
				#fisher[i,:] = l2_normalize_vector(fisher[i,:])

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
		test_images_desc = folds_descriptors[fold_i]['descriptors']
		test_labels = folds_descriptors[fold_i]['label_per_descriptor']

		test_images_desc = np.asarray(test_images_desc)

		fisher_test=np.zeros((len(test_images_desc),k*D.shape[1]*2*test_images_desc.shape[1]),dtype=np.float32)
		for i in range(len(test_images_desc)):
			for j in range(test_images_desc.shape[1]): #number of levels
				des = test_images_desc[i][j] # now only working with 1 PYRAMID LEVEL [0]
				if reduction == 'pca':
					des = pca_reducer.transform(des)
				fisher_test[i,j*k*D.shape[1]*2:(j+1)*k*D.shape[1]*2]=ynumpy.fisher(gmm, np.float32(des), include = ['mu','sigma'])
				#fisher_test[i,:] = l2_normalize_vector(fisher_test[i,:])

		fisher_test = stdSlr.transform(fisher_test)
		#fisher_test = l2_normalize_vector(fisher_test)
		if kernel == 'pyramid_match':
			predictMatrix = spatialPyramidKernel(fisher_test, D_scaled, k*D.shape[1]*2, pyramid)
			#predictions = clf.predict(predictMatrix)
			#predictions_proba = clf.predict_proba(predictMatrix)
			accuracy = 100*clf.score(predictMatrix, test_labels)
		else:
			accuracy = 100*clf.score(fisher_test, test_labels)

		print 'Fold '+str(fold_i)+' accuracy: ' + str(accuracy)

		accuracies.append(accuracy)

	return np.asarray(accuracies)

def main(nfeatures=100, code_size=512, n_components=60, kernel='linear', C=1, reduction=None, features='sift', pyramid=None, grid_step=6):
	start = time.time()

	# read the train and test files
	folds_data = get_cross_val_dataset()

	# create the SIFT detector object
	cnn_model = cnn_features('fc2')

	# read all the images per train
	# extract SIFT keypoints and descriptors
	# store descriptors in a python list of numpy arrays
	folds_descriptors = getFoldsDescriptors(cnn_model, folds_data, pyramid)

	# now perform de cross-val
	folds_num = 5
	accuracies = getCrossVal(folds_num, folds_descriptors, start, nfeatures, code_size,  kernel, C,features, pyramid, grid_step, n_components, reduction)

	print('Final accuracy: %f (%f)' % (np.mean(accuracies), np.std(accuracies)))

parser = argparse.ArgumentParser()
parser.add_argument('-n_feat', help='Number of features per image to use', type=int, default=100)
parser.add_argument('-code_size', help='Codebook size', type=int, default=32)
parser.add_argument('-n_comp', help='Number of features to keep after feature reduction', type=int, default=60)
parser.add_argument('-kern', help='SVM kernel to use', type=str, default='linear')
parser.add_argument('-C', help='SVM C parameter', type=float, default=1.0)
parser.add_argument('-reduce', help='Feature reduction', type=str, default=None)
parser.add_argument('-feats', help='Features to use', type=str, default='dense_sift')
parser.add_argument('-grid_step', help='step of the sift grid', type=int, default=6)
args = parser.parse_args()

print(args)

main(args.n_feat, args.code_size, args.n_comp, args.kern, args.C, args.reduce, args.feats, args.pyramid, args.grid_step)