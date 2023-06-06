import numpy as np
from numpy import random
from scipy import sparse
import scipy.io as sio
import time
import math

if __name__ == "__main__":
    '''
        n 		= size of each block
        k 		= # of blocks
        alpha_i	= parameter in the differential equations of i-th block
        Annotation:
            the corrosponding d is n + 1
    '''
    n = 4
    k = int(3e6)
    N = n * k
    # non_zero = k * (n + 2 * (n-1)) + 2 * (k-1) * n
    non_zero = k * (n + (n - 1)) + (k - 1) * n

    with open('/public/home/LeiChao/Document/MDMRM/data/testdata.mtx', 'w') as f:
        f.write('%%MatrixMarket matrix coordinate real symmetric\n')
        f.write('{} {} {}\n'.format(N, N, non_zero))

        # generate every block, row is based 1
        for i in range(k):
            alpha = random.randint(5, 300)
            for row in range(i * n + 1, (i+1) * n + 1):
                # left identity
                if(row - n >= 1):
                    f.write('{} {} {}\n'.format(row, row - n, -1))
                # block row
                if(row != i * n + 1):
                    f.write('{} {} {}\n'.format(row, row - 1, -1))
                f.write('{} {} {}\n'.format(row, row, 4 - pow(alpha, 2)))
                '''
                if(row != (i+1) * n):
                    f.write('{} {} {}\n'.format(row, row + 1, -1))
                '''

                '''
                # right identity
                if(row + n <= N):
                    f.write('{} {} {}\n'.format(row, row + n, -1))
                '''