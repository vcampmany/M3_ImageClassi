import os.path
import time
import numpy as np
import cPickle
from yael import ynumpy

def compute_codebook(D, code_size, nfeatures, fold_i=None, features='sift'):
	if features == 'sift':
		features = '' # do not change filename for basic sift
	elif features == 'dense_sift':
		features = 'dense_sift_'

	if fold_i is not None:
		code_name = "codebooks/"+str(code_size)+"_"+features+str(nfeatures)+"_fold_"+str(fold_i)+".dat"
	else:
		code_name = "codebooks/"+str(code_size)+"_"+features+str(nfeatures)+".dat"
	if not os.path.isfile(code_name):
		print 'Computing kmeans with '+str(code_size)+' centroids'
		init=time.time()
		codebook = ynumpy.gmm_learn(np.float32(D), code_size)
		cPickle.dump(codebook, open(code_name, "wb"))
		end=time.time()
		print 'Done in '+str(end-init)+' secs.'
	else:
		codebook = cPickle.load(open(code_name, "r"))

	return codebook