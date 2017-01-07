import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm

def histogram_intersection(m, n):
	n_samples, dim = m.shape
	K_int = np.zeros((n_samples,n_samples))

	for i in range(n_samples):
		for j in range(n_samples):
			K_int[i,j] = np.minimum(m[i,:], m[j,:]).sum()

	return K_int.reshape((n_samples*n_samples,1))
	# TODO not finished

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
    K = np.minimum(X[:, np.newaxis, :],
                   Y[np.newaxis, :, :]).sum(axis=2)
    return K


def histogram_intersection_kernel(train, test, C, GT_train):
	stdSlr = StandardScaler().fit(train)
	train_scaled = stdSlr.transform(train)
	kernelMatrix = histogram_intersection(train_scaled, train_scaled)
	clf = svm.SVC(kernel='precomputed', C=C)
	clf.fit(kernelMatrix, GT_train)
	predictMatrix = histogram_intersection(stdSlr.transform(test), train_scaled)
	SVMpredictions = clf.predict(predictMatrix)