#!/usr/bin/python

#Benchmark version (original matlab code ported to python)
#This version uses CVXPy 1.0
#The goal is to demonstrate that MOSEK has a memory leak.
#In this example, a rank-1 matrix is recovered from dimension deficient data

import time
import numpy as np
import cvxpy

print "CVXPY version: ", cvxpy.__version__

#Initialize
np.random.seed(1337)

#matrix dimension
n = 60

#number of tests
reps = 1

#List contains largest deviation for each rep
fmat = []

#Choose solver
s = cvxpy.SCS

#Wallclock time
start = time.time()

for j in range(0, reps):
	#Draw rank-1 matrix and vectorize
	M0 = np.matmul(np.random.randn(n,1), np.matrix.transpose(np.random.randn(n,1)))
	vecmat = M0.reshape(n * n, order='F')
		
	#Draw linear map
	A = np.random.randn(6*n, n*n)
	
	#y = incomplete information about M0	
	y = np.matmul(A, vecmat)
	
	#Optimize
	X = cvxpy.Variable((n, n))
	Y = cvxpy.Variable((n, n))
	M = cvxpy.Variable((n, n))
	Z = cvxpy.Variable((2*n, 2*n), PSD=True)

	objective = cvxpy.Minimize(cvxpy.trace(X+Y)/2)
	constr1 = Z == cvxpy.bmat([[cvxpy.multiply(0.5, (X+X.T)), M], [M.T, cvxpy.multiply(0.5, (Y+Y.T))]])
	constr2 = y == A * cvxpy.vec(M)
	constraints = [constr1, constr2]
	
	prob = cvxpy.Problem(objective, constraints)
	
	#Wallclock time
	start_inner = time.time()
	prob.solve(solver=s, verbose=True)
	end_inner = time.time()
	print("inner loop wallclock time = " + str(end_inner - start_inner))
	
	#Get results
	M1 = M.value	
	fmat.append(np.linalg.norm(M1.reshape(n * n, order='F') - vecmat))
	print prob.status	

	#print "j = ", j, " first row of M0:\n", M0[0]
	#print("\nj = " + str(j) + " first row of M1:\n" + str(M1[0]) + "\n")

end = time.time()
print("outer loop wallclock time = " + str(end - start))

print(fmat)
largest = np.max(np.absolute(fmat))
print("largest deviation = " + str(largest))

