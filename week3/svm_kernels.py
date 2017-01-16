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

	pyramid = np.concatenate(([1,1], pyramid))

	levels = len(pyramid) / 2
	# index where each level of the pyramid starts
	startOffsets = np.zeros(levels, dtype=np.int)
	#startOffsets[0] = 0
	for i in range(len(startOffsets)):
		for j in range(i):
			startOffsets[i] += pyramid[j*2] * pyramid[j*2+1] * histDim

	sizes = np.zeros(levels, dtype=np.int)
	for i in range(len(sizes)):
		sizes[i] = pyramid[i*2] * pyramid[i*2+1] * histDim
	
	
	aux = histogramIntersectionKernel(X[:,0:histDim], Y[:,0:histDim])
	ite = np.zeros((n_samples_1,n_samples_2))

	#print 'startOffsets', startOffsets
	#print 'sizes', sizes
	for l in range(1,levels):
		ite += 2**(-l) * ( histogramIntersectionKernel( X[:, startOffsets[l]:startOffsets[l]+sizes[l]], \
									             Y[:, startOffsets[l]:startOffsets[l]+sizes[l]]) - \
					histogramIntersectionKernel( X[:, startOffsets[l-1]:startOffsets[l-1]+sizes[l-1]], \
									             Y[:, startOffsets[l-1]:startOffsets[l-1]+sizes[l-1]]) )
#		#print 'non zero', np.count_nonzero(ite)
#		#print ite.shape

	#aux = (1 / 2**levels-1)*histogramIntersectionKernel(X[:,0:histDim], Y[:,0:histDim])
	#for l in range(1,levels):
	#	ite += 1/2**(levels-l+1) * histogramIntersectionKernel( X[:, startOffsets[l]:startOffsets[l]+sizes[l]], \
	#								             Y[:, startOffsets[l]:startOffsets[l]+sizes[l]] )

	ker = aux + ite
	return ker