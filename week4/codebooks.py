import os.path
import time
import numpy as np
import cPickle
from yael import ynumpy

def compute_codebook(D, code_size, nfeatures, fold_i=None, output='fc2', n_comp=128):
	
	features = 'cnn_'+output

	if fold_i is not None:
		code_name = "codebooks/"+str(code_size)+"_"+str(n_comp)+"_"+features+"_fold_"+str(fold_i)+".dat"
	else:
		code_name = "codebooks/"+str(code_size)+"_"+str(n_comp)+"_"+features+".dat"
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