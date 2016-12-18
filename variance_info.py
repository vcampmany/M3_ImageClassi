'''Loads variance_explained.npy to return the amount of information for a given number of dimensions'''

import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-ndim', help='Number of dimensions', type=int, default=20)
args = parser.parse_args() 

var_explained = np.load('variance_explained.npy')

print(var_explained.shape)
print(var_explained[args.ndim])
