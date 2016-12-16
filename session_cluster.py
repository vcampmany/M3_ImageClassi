import cv2
import numpy as np
import cPickle
import time
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.decomposition import PCA
import sys

def get_dataset():
	train_images_filenames = cPickle.load(open('train_images_filenames.dat','r'))
	train_images_filenames = [filename.replace('../../Databases/', '') for filename in train_images_filenames]
	test_images_filenames = cPickle.load(open('test_images_filenames.dat','r'))
	test_images_filenames = [filename.replace('../../Databases/', '') for filename in test_images_filenames]
	train_labels = cPickle.load(open('train_labels.dat','r'))
	test_labels = cPickle.load(open('test_labels.dat','r'))

	print 'Loaded '+str(len(train_images_filenames))+' training images filenames with classes ',set(train_labels)
	print 'Loaded '+str(len(test_images_filenames))+' testing images filenames with classes ',set(test_labels)

	return train_images_filenames, test_images_filenames, train_labels, test_labels

def get_feature_detector(name='sift', n_features=100):
	if name == 'sift':
		return cv2.SIFT(nfeatures=n_features)
	elif name == 'orb':
		return cv2.ORB(nfeatures=n_features)
	elif name == 'surf':
		return cv2.SURF(hessianThreshold=400)
	else:
		raise NotImplemented

def extract_features(FEATdetector, train_images_filenames, train_labels, nImages):
	Train_descriptors = []
	Train_label_per_descriptor = []

	for i in range(len(train_images_filenames)):
		filename=train_images_filenames[i]
		if Train_label_per_descriptor.count(train_labels[i])<nImages:
			print 'Reading image '+filename
			ima=cv2.imread(filename)
			gray=cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
			kpt,des=FEATdetector.detectAndCompute(gray,None)
			Train_descriptors.append(des)
			Train_label_per_descriptor.append(train_labels[i])
			print str(len(kpt))+' extracted keypoints and descriptors'

	# Transform everything to numpy arrays

	D=Train_descriptors[0]
	L=np.array([Train_label_per_descriptor[0]]*Train_descriptors[0].shape[0])

	for i in range(1,len(Train_descriptors)):
		D=np.vstack((D,Train_descriptors[i]))
		L=np.hstack((L,np.array([Train_label_per_descriptor[i]]*Train_descriptors[i].shape[0])))

	return D, L

def train_SVM(kernel, C, D, L):
	stdSlr = StandardScaler().fit(D)
	D_scaled = stdSlr.transform(D)
	print 'Training the SVM classifier...'
	clf = svm.SVC(kernel='linear', C=1).fit(D_scaled, L)
	print 'Done!'

	return clf, stdSlr

def test_SVM(FEATdetector, test_images_filenames, test_labels, clf, stdSlr, reducer):
	numtestimages=0
	numcorrect=0
	for i in range(len(test_images_filenames)):
		filename=test_images_filenames[i]
		ima=cv2.imread(filename)
		gray=cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
		kpt,des=FEATdetector.detectAndCompute(gray,None)
		if reducer:
			des = reducer.transform(des)
		predictions = clf.predict(stdSlr.transform(des))
		values = np.unique(predictions)
		counts = np.bincount(predictions)
		predictedclass = values[np.argmax(counts)]
		print 'image '+filename+' was from class '+test_labels[i]+' and was predicted '+predictedclass
		numtestimages+=1
		if predictedclass==test_labels[i]:
			numcorrect+=1

	return numcorrect, numtestimages

def PCA_reduce(D, n_components):
	print(D.shape)
	pca = PCA(n_components=n_components)
	pca.fit(D)
	return pca.transform(D), pca

def main(nfeatures=100, nImages=30, n_components=20, kernel='linear', C=1, reduction=None, features='sift', outFile='out'):
	start = time.time()

	# read the train and test files
	train_images_filenames, test_images_filenames, train_labels, test_labels = get_dataset()

	# create the SIFT detector object
	FEATdetector = get_feature_detector(name=features, n_features=nfeatures)

	# read the just 30 train images per class
	# extract SIFT keypoints and descriptors
	# store descriptors in a python list of numpy arrays
	D, L = extract_features(FEATdetector, train_images_filenames, train_labels, nImages)

	if reduction == 'pca':
		D, reducer = PCA_reduce(D, n_components)
	else:
		reducer = None

	print(D.shape)
	# Train a linear SVM classifier
	clf, stdSlr = train_SVM(kernel, C, D, L)

	# get all the test data and predict their labels
	numcorrect, numtestimages = test_SVM(FEATdetector, test_images_filenames, test_labels, clf, stdSlr, reducer)

	print 'Final accuracy: ' + str(numcorrect*100.0/numtestimages)

	end=time.time()
	print 'Done in '+str(end-start)+' secs.'

	f = open(outFile,'a')
	f.write('Config= nfeatures:'+nfeatures+' nImages:'+nImages+' n_components:'+n_components+' kernel:'+kernel+ ' c:'+c+ ' reduction:'+reduction+' features:'+features+ '   Final accuracy= ' + str(numcorrect*100.0/numtestimages) + ' Done in '+str(end-start)+' secs.' )
	f.close()

## 38.78% in 797 secs.


nfeatures = int(sys.argv[1])
nImages = int(sys.argv[2])
n_components = int(sys.argv[3])
kernel = str(sys.argv[4])
c = int(sys.argv[5])
reduction = str(sys.argv[6])
features = str(sys.argv[7])
outFile = str(sys.argv[8])

main(nfeatures, nImages, n_components, kernel, c, reduction, features, outFile)