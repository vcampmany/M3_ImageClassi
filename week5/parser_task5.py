import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("path", type=str, default="", nargs="?")
args = parser.parse_args()

LOGS = './task5_results/'+args.path

results = os.listdir(LOGS)

results = [res for res in results if 'run1_' in res]

n_results = len(results)-1

accuracies = []
for i in range(n_results):
	result = 'run1_'+str(i)+'.txt';
	print(result)
	with open(LOGS+result) as f:
		lines = f.readlines()
		for line in lines:
			if 'Namespace' in line:
				params = line.rstrip()
		test_acc = lines[-1].rstrip().split(' ')[-1]
		if test_acc != 'KeyboardInterrupt' and 'Terminated' not in test_acc:
			accuracies.append(float(test_acc))
			if float(test_acc) > 0.94:
				print(test_acc)
				print(params)

print('Max accuracy until now: %f' % max(accuracies))