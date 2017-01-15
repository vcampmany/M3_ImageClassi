import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm

def histogramIntersectionKernel(X, Y):
	"""
	Compute the histogram intersection kernel (min kernel)
	between X and Y::
	    K(x, y) = \sum_i^n min(|x_i|^\alpha, |y_i|^\beta)
	Parameters
	----------
	X : array of shape (n_samples_1, n_features)
	Y : array of shape (n_samples_2, n_features)
	Returns
	-------
	Gram matrix : array of shape (n_samples_1, n_samples_2)
	"""
	n_samples_1, n_features = X.shape
	n_samples_2, _ = Y.shape
	# K = np.minimum(X[:, np.newaxis, :],
	#                Y[np.newaxis, :, :]).sum(axis=2)

	K = np.zeros((n_samples_1,n_samples_2))

	for i in range(n_samples_1):
		for j in range(n_samples_2):
			K[i,j] = np.minimum(X[i,:], Y[j,:]).sum()

	return K


def spatialPyramidKernel(X, Y, histDim, pyramid):

	n_samples_1, n_features = X.shape
	n_samples_2, _ = Y.shape

	levels = len(pyramid) / 2
	# index where each level of the pyramid starts
	startOffsets = np.zeros(levels, dtype=np.int)
	for i in range(len(startOffsets)):
		for j in range(i-1):
			startOffsets[i] += pyramid[j*2] + pyramid[j*2+1]

	# index where each level of the pyramid ends
	endOffsets = np.zeros(levels, dtype=np.int)
	for i in range(len(startOffsets)):
		for j in range(i):
			endOffsets[i] += pyramid[j*2] + pyramid[j*2+1]
	
	#aux = (1 / 2**levels)*histogramIntersectionKernel(X[:,0:histDim], Y[:,0:histDim])
	aux = histogramIntersectionKernel(X[:,0:histDim], Y[:,0:histDim])
	ite = np.zeros((n_samples_1,n_samples_2))

	for l in range(1,levels):
		ite += 2**(-l) * ( histogramIntersectionKernel( X[:, startOffsets[l]*histDim:endOffsets[l]*histDim], \
									             Y[:, startOffsets[l]*histDim:endOffsets[l]*histDim]) - \
					histogramIntersectionKernel( X[:, startOffsets[l-1]*histDim:endOffsets[l-1]*histDim], \
									             Y[:, startOffsets[l-1]*histDim:endOffsets[l-1]*histDim]) )  \
	
	ker = aux + ite
	return ker