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