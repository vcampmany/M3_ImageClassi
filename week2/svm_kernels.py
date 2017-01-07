import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm

def histogram_intersection(m, n):
	return np.minimum(m,n).sum()
	# TODO not finished


def histogram_intersection_kernel(train, test, C, GT_train):
	stdSlr = StandardScaler().fit(train)
	train_scaled = stdSlr.transform(train)
	kernelMatrix = histogram_intersection(train_scaled, train_scaled)
	clf = svm.SVC(kernel='precomputed', C=C)
	clf.fit(kernelMatrix, GT_train)
	predictMatrix = histogram_intersection(stdSlr.transform(test), train_scaled)
	SVMpredictions = clf.predict(predictMatrix)