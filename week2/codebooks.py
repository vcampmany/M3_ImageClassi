import os.path
import cv2
import time
from utils import get_dataset
import numpy as np
from sklearn import cluster
import cPickle

def compute_codebook(D, code_size, nfeatures, fold_i=None):
	if fold_i is not None:
		code_name = "codebooks/"+str(code_size)+"_"+str(nfeatures)+"_fold_"+str(fold_i)+".dat"
	else:
		code_name = "codebooks/"+str(code_size)+"_"+str(nfeatures)+".dat"
	if not os.path.isfile(code_name):
		print 'Computing kmeans with '+str(code_size)+' centroids'
		init=time.time()
		codebook = cluster.MiniBatchKMeans(n_clusters=code_size, verbose=False, batch_size=code_size * 20,compute_labels=False,reassignment_ratio=10**-4)
		codebook.fit(D)
		cPickle.dump(codebook, open(code_name, "wb"))
		end=time.time()
		print 'Done in '+str(end-init)+' secs.'
	else:
		codebook = cPickle.load(open(code_name, "r"))

	return codebook

def main(nfeatures, code_size):
	# read the train and test files
	train_images_filenames, test_images_filenames, train_labels, test_labels = get_dataset(verbose=False)

	# create the SIFT detector object
	SIFTdetector = cv2.SIFT(nfeatures=nfeatures)

	# read all the images per train
	# extract SIFT keypoints and descriptors
	# store descriptors in a python list of numpy arrays

	Train_descriptors = []
	Train_label_per_descriptor = []

	print("Computing sift descriptors")
	for i in range(len(train_images_filenames)):
		filename=train_images_filenames[i]
		#print 'Reading image '+filename
		ima=cv2.imread(filename)
		gray=cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
		kpt,des=SIFTdetector.detectAndCompute(gray,None)
		Train_descriptors.append(des)
		Train_label_per_descriptor.append(train_labels[i])
		#print str(len(kpt))+' extracted keypoints and descriptors'

	# Transform everything to numpy arrays
	size_descriptors=Train_descriptors[0].shape[1]
	D=np.zeros((np.sum([len(p) for p in Train_descriptors]),size_descriptors),dtype=np.uint8)
	startingpoint=0
	for i in range(len(Train_descriptors)):
		D[startingpoint:startingpoint+len(Train_descriptors[i])]=Train_descriptors[i]
		startingpoint+=len(Train_descriptors[i])

	# Compute Codebook
	codebook = compute_codebook(D, code_size, nfeatures)

if __name__ == '__main__':
	params = [
		(100,4086),
		(500,2048),
		(500,4086),
	]

	for param in params:
		print("Computing codebook: n_feat %i, code_size: %i" % (param[0], param[1]))
		main(param[0], param[1])