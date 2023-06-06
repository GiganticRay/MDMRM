import numpy as np
from numpy import random
from scipy import sparse
import scipy.io as sio
import time
import math

if __name__ == "__main__":
	N = 2 * int(1e3)
	d = 5
	m = 2 * d - 1
	
	uniform_lb = 1 
	uniform_rb = 2

	# 生成原理：先生成 L, 保证 L 的对角线元素全 > 0, 且 L 是 下 baneded_matrix format, 然后生成 LLT, 就是正定的带状矩阵了
	mtx = np.eye(N)

	max_elem = -math.inf
	min_elem = math.inf
	for i in range(N):
		lb = max(0, i - d + 1)
		rb = i
		mtx[i][lb : rb] = [random.randint(1, 10) 
								for i in range(rb - lb)]

		mtx[i][i] = random.randint(1, 1e7)
		max_elem = max(max_elem, mtx[i][i])
		min_elem = min(min_elem, mtx[i][i])

	print("{}, {}, {:e}".format(max_elem, min_elem, pow(max_elem/min_elem, 2)))
	mtx = np.dot(mtx, np.transpose(mtx))

	# convert into scipy sparse
	sp_mtx = sparse.coo_matrix(mtx)

	# save into matrix market format
	sio.mmwrite("/public/home/LeiChao/Document/MDMRM/data/self_generated_{}_{}.mtx".format(d, N), sp_mtx)
	# sio.mmwrite("/public/home/LeiChao/Document/MDMRM/data/EIGENTEST_{}_{}.mtx".format(d, N), sp_mtx)


