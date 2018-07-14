#####################################################################################
#																					#
#	 This is a script for evaluating estimations. Usage is given in the script.     #
#																					#
#####################################################################################

import numpy as np
import sys

def main(argv):
	if len(argv) != 2:
		print("Usage: python evaluate.py estimations.npy groundtruth.npy")
		exit()
	
	y_hat = np.load(argv[0])
	y = np.load(argv[1])
	
	if y_hat.size != y.size:
		print("Shape mismatch between estimations and ground truth")
		print("Estimation: ", y_hat.size)
		print("Ground truth: ", y.size)
		exit()
		
	y_hat = y_hat.reshape(-1,1)
	y = y.reshape(-1,1)
	
	print((np.abs(y_hat-y) < 10).sum() / y.shape[0])

if __name__ == "__main__":
   main(sys.argv[1:])
