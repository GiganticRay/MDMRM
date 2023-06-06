#include "CUDAKernel.cuh"

#define GRID_SIZE 	256
#define BLOCK_SIZE 	256

// kernel function
__global__ void cu_dot(double *v1, double *v2, double *res, int n){
	int index = threadIdx.x + blockDim.x*blockIdx.x;
	int stride = blockDim.x*gridDim.x;

	// error: expression must have a constant value
	// sahread memory is shared inside block
	__shared__ double cache[BLOCK_SIZE];
	double temp = 0.0;

	while(index < n){
		temp += v1[index] * v2[index];
		index += stride;
	}

	cache[threadIdx.x] = temp;
	__syncthreads();

	// reduction
	int i = blockDim.x/2;
	while(i != 0){
		if(threadIdx.x < i){
			cache[threadIdx.x] += cache[threadIdx.x + i];
		}
		__syncthreads();
		i /= 2;
	}
	if(threadIdx.x == 0){
		atomicAdd(res, cache[0]);
	}
}

__global__ void cu_BM_V_MULT(double *BM_A, double *x, double *res, size_t N, size_t d){
	int index = threadIdx.x + blockDim.x*blockIdx.x;
	int stride = blockDim.x*gridDim.x;

	while(index < N){
		// 处理第 index 行的数据
		int row_idx = index + 1;
		int lb	= max(1, int(row_idx - d + 1));
		int ub	= min(N, row_idx + d - 1);
		double item = 0;
		for(int col_idx = lb; col_idx <= ub; col_idx++){
			int row_in_BM = d - 1 + 1 + row_idx - col_idx;
			int col_in_BM = col_idx;
			item += BM_A[(row_in_BM - 1) * N + col_in_BM - 1] * x[col_idx - 1];
		}
		res[row_idx - 1] = item;
		index += stride;
	}
}

__global__ void cu_compute_H(double *BM_A, double *r, double *H, int N, int d, int l){
	int index = threadIdx.x + blockDim.x*blockIdx.x;
	int stride = blockDim.x*gridDim.x;
	int m = 2 * d - 1;

	while(index < N){
		// 处理 H[i]
		int x = index + 1;
		int lb	= max(1, int(x - d + 1));
		int ub	= min(N, x + d - 1);
		for(int y = lb; y <= ub; y++){
			if((y - l) % m == 0)
				continue;
			else{
				int s_idx = (y - 1) % m + 1;
				// 坐标转化, A[row_in_A][col_in_A] -> BM_A[row_in_BM_A][col_in_BM_A] -> BM[***]
				int row_in_A = x;
				int col_in_A = y;
				int row_in_BM_A = d - 1 + 1 + row_in_A - col_in_A;
				int col_in_BM_A = col_in_A;
				int coord_in_BM_A = (row_in_BM_A - 1) * N + col_in_BM_A;

				int x_in_H = x;
				int y_in_H = s_idx + (s_idx > l ? -1 : 0);
				int coord_in_H = (x_in_H - 1) * (m - 1) + y_in_H;
				H[coord_in_H - 1] = BM_A[coord_in_BM_A - 1] * r[y-1];
			}
		}
		index += stride;
	}
}

__global__ void cu_compute_STEP2(double *BM_A, double *r, double *H, double *u, double *z, double *y, double *q, int N, int t, int d, int l){
	int index = threadIdx.x + blockDim.x*blockIdx.x;
	int stride = blockDim.x*gridDim.x;
	int m = 2 * d - 1;

	while(index < t){
		int x = index + 1;
		int off_set_idx = m * (x - 1) + l;
		int lb	= max(1, int(off_set_idx - d + 1));
		int ub	= min(N, off_set_idx + d - 1);

		for(int j = lb; j <= ub; j++){
			// 索引 A[j - 1][off_set_idx - 1]
			int row_in_Dense = j;
			int col_in_Dense = off_set_idx;	
			int row_in_BM_A = d - 1 + 1 + row_in_Dense - col_in_Dense;
			int col_in_BM_A = col_in_Dense;
			int coord_in_BM_A = (row_in_BM_A - 1) * N + col_in_BM_A;

			u[x - 1] += r[j - 1] * BM_A[coord_in_BM_A - 1];
			z[x - 1] += BM_A[coord_in_BM_A - 1] * BM_A[coord_in_BM_A - 1];

			int h_lb = (j - 1) * (m - 1) + 1;
			int y_q_lb = (x - 1) * (m - 1) + 1;
			for(int i = 0; i < m - 1; i++){
				y[y_q_lb - 1 + i] += BM_A[coord_in_BM_A - 1] * H[h_lb - 1 + i];
				q[y_q_lb - 1 + i] += BM_A[coord_in_BM_A - 1] * r[off_set_idx - 1] * H[h_lb - 1 + i];
			}
		}

		index += stride;
	}
}

// A 是行排列，B 是列排列
// 每个元素计算用 reduction 策略
__global__ void cu_MM_LElem_MComp(double *A, double *B, double *R, int m, int n, int k){
	int index = threadIdx.x + blockDim.x*blockIdx.x;
	int stride = blockDim.x*gridDim.x;

	while(index < m * k){
		int idx = index + 1;
		int row_idx = (idx - 1) / k + 1;
		int col_idx = (idx - 1) % k + 1;
		int coord_in_R = (row_idx - 1) * k + col_idx;
		
		double* row_base_A = A + (row_idx - 1) * n;
		double* col_base_B = B + (col_idx - 1) * n;
		cu_dot<<<GRID_SIZE, BLOCK_SIZE>>>(row_base_A, col_base_B, &R[coord_in_R - 1], n);

		index += stride;
	}
}

// 每个元素直接计算
__global__ void cu_MM_MElem_LComp(double *A, double *B, double *R, int m, int n, int k){
	int index = threadIdx.x + blockDim.x*blockIdx.x;
	int stride = blockDim.x*gridDim.x;

	while(index < m * k){
		int idx = index + 1;
		int row_idx = (idx - 1) / k + 1;
		int col_idx = (idx - 1) % k + 1;
		int coord_in_R = (row_idx - 1) * k + col_idx;
		
		int row_base_A_idx = (row_idx - 1) * n + 1;
		int col_base_B_idx = (col_idx - 1) * n + 1;
		for(int i = 0; i < n; i++){
			R[coord_in_R - 1] += A[row_base_A_idx - 1 + i] * B[col_base_B_idx - 1 + i]; 
		}	

		index += stride;
	}
}

__global__ void cu_compute_e1_i(double *H_i, double *r, double *e1_i, int N, int m){
	int index = threadIdx.x + blockDim.x*blockIdx.x;
	int stride = blockDim.x*gridDim.x;

	__shared__ double cache[BLOCK_SIZE];
	double temp = 0.0;

	while(index < N){
		int idx_in_loop = index + 1;
		temp += r[idx_in_loop - 1] * H_i[idx_in_loop - 1];
		index += stride;
	}

	cache[threadIdx.x] = temp;
	__syncthreads();

	// reduction
	int i = blockDim.x/2;
	while(i != 0){
		if(threadIdx.x < i){
			cache[threadIdx.x] += cache[threadIdx.x + i];
		}
		__syncthreads();
		i /= 2;
	}

	if(threadIdx.x == 0){
		atomicAdd(e1_i, cache[0]);
	}
}

__global__ void cu_copmute_rbM_inverse_vec(double *z, double *r, double *res, int N, int m, int t, int l){
	int index = threadIdx.x + blockDim.x*blockIdx.x;
	int stride = blockDim.x*gridDim.x;

	while(index < t){
		int x = index + 1;
		if(z[x - 1] * r[(x - 1) * m + l - 1] == 0)
			res[x - 1] = 1;
		res[x - 1] = 1.0 / (z[x - 1] * r[(x - 1) * m + l - 1]);
		index += stride;
	}
}

__global__ void cu_RowTrans(double* coeff_vec, double *mtx, double *res, int m, int n){
	int index = threadIdx.x + blockDim.x*blockIdx.x;
	int stride = blockDim.x*gridDim.x;

	while(index < m * n){
		int idx = index + 1;	
		int row_idx = (idx - 1) % m + 1;
		res[idx - 1] = mtx[idx - 1] * coeff_vec[row_idx - 1];

		index += stride; 
	}
}

__global__ void cu_RestoreD(double *s, double *alpha, double *D, int N, int d, int l, int m, int t){
	int index = threadIdx.x + blockDim.x*blockIdx.x;
	int stride = blockDim.x*gridDim.x;

	while(index < t + 1){
		int idx = 0;
		for(int j =  index * m; j < (index+1) * m && j < N; j++){
			int x = j + 1;
			if((x - l) % m == 0)
				D[x-1] = alpha[index];
			else
				D[x-1] = s[idx++];
		}
		index += stride;
	}
}

// without cache optimization
// https://stackoverflow.com/questions/16737298/what-is-the-fastest-way-to-transpose-a-matrix-in-c
__global__ void cu_GetTranspose(double *src, double *dst, int m, int n){
	int index = threadIdx.x + blockDim.x*blockIdx.x;
	int stride = blockDim.x*gridDim.x;

	while(index < m * n){
		int row_in_src = index % m;	// base 0
		int col_in_src = index / m;
		int coord_in_dst = n * row_in_src + col_in_src;
		dst[coord_in_dst] = src[index];
		index += stride;
	}
}

__global__ void cu_Add(double *vec1, double *vec2, double *res, int n){
	int index = threadIdx.x + blockDim.x*blockIdx.x;
	int stride = blockDim.x*gridDim.x;

	while(index < n){
		res[index] = vec1[index] + vec2[index];
		index += stride;
	}
}

__global__ void cu_Sub(double *vec1, double *vec2, double *res, int n){
	int index = threadIdx.x + blockDim.x*blockIdx.x;
	int stride = blockDim.x*gridDim.x;

	while(index < n){
		res[index] = vec1[index] - vec2[index];
		index += stride;
	}
}

__global__ void cu_Scalar(double *vec1, double *res, double scalar, int n){
	int index = threadIdx.x + blockDim.x*blockIdx.x;
	int stride = blockDim.x*gridDim.x;

	while(index < n){
		res[index] = scalar * vec1[index];
		index += stride;
	}
}


// wrapper
// data memory logic
double dot(const Eigen::VectorXd & v1, const Eigen::VectorXd & v2){
	int n = v1.size();
	int size_in_bytes = n * sizeof(double);
	double *dev_v1, *dev_v2;	
	double res, *dev_res;
	// STEP 1: Allocate device memory
	cudaMalloc((void**) &dev_v1, size_in_bytes);
	cudaMalloc((void**) &dev_v2, size_in_bytes);
	cudaMalloc((void**) &dev_res, sizeof(double));
	cudaMemset(dev_res, 0, sizeof(double));

	cudaMemcpy(dev_v1, v1.data(), size_in_bytes, cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy(dev_v2, v2.data(), size_in_bytes, cudaMemcpyKind::cudaMemcpyHostToDevice);

	// STEP 2: Invoke kernel to compute dot 
	cu_dot<<<GRID_SIZE, BLOCK_SIZE>>>(dev_v1, dev_v2, dev_res, n);
	cudaDeviceSynchronize();

	// STEP 3: Copy data back to host
	cudaMemcpy(&res, dev_res, sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost);

	// STEP 4: Free device memory
	cudaFree(dev_v1);
	cudaFree(dev_v2);
	cudaFree(dev_res);

	return res;
}
double dot(double* v1, double *v2, int n){
	int size_in_bytes = n * sizeof(double);
	double *dev_v1, *dev_v2;	
	double res, *dev_res;
	// STEP 1: Allocate device memory
	cudaMalloc((void**) &dev_v1, size_in_bytes);
	cudaMalloc((void**) &dev_v2, size_in_bytes);
	cudaMalloc((void**) &dev_res, sizeof(double));
	cudaMemset(dev_res, 0, sizeof(double));

	cudaMemcpy(dev_v1, v1, size_in_bytes, cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy(dev_v2, v2, size_in_bytes, cudaMemcpyKind::cudaMemcpyHostToDevice);

	// STEP 2: Invoke kernel to compute dot 
	cu_dot<<<GRID_SIZE, BLOCK_SIZE>>>(dev_v1, dev_v2, dev_res, n);
	cudaDeviceSynchronize();

	// STEP 3: Copy data back to host
	cudaMemcpy(&res, dev_res, sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost);

	// STEP 4: Free device memory
	cudaFree(dev_v1);
	cudaFree(dev_v2);
	cudaFree(dev_res);

	return res;
}

Eigen::VectorXd BM_V_MULT(vector<double> BM_A, int N, int d, Eigen::VectorXd x){
	double *dev_BM_A_p, *dev_x, *dev_res, *res;
	
	// STEP 1: Allocate device memory and host memory
	cudaMalloc((void**) &dev_BM_A_p, BM_A.size() * sizeof(double));
	cudaMalloc((void**) &dev_x,	N * sizeof(double));
	cudaMalloc((void**) &dev_res, N * sizeof(double));
	res = (double*) malloc (N * sizeof(double));

	cudaMemset(dev_res, 0, N * sizeof(double));
	memset(res, 0, N * sizeof(double));

	cudaMemcpy(dev_BM_A_p, BM_A.data(), BM_A.size() * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy(dev_x, x.data(), N * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);

	// STEP 2: Invoke kernel to compute res
	cu_BM_V_MULT<<<GRID_SIZE, BLOCK_SIZE>>>(dev_BM_A_p, dev_x, dev_res, N, d);
	cudaDeviceSynchronize();

	// STEP 3: Copy data back to host
	cudaMemcpy(res, dev_res, N * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost);
	Eigen::VectorXd res_vec = Eigen::Map<Eigen::VectorXd>(res, N);

	// STEP 4: Free Device memory and host memory
	cudaFree(dev_BM_A_p);
	cudaFree(dev_x);
	cudaFree(dev_res);
	free(res);	

	return res_vec;
}
void BM_V_MULT(double *BM_A, int BM_A_size, double *x, double *res, int N, int d){
	double *dev_BM_A_p, *dev_x, *dev_res;
	
	// STEP 1: Allocate device memory and host memory
	cudaMalloc((void**) &dev_BM_A_p, BM_A_size * sizeof(double));
	cudaMalloc((void**) &dev_x,	N * sizeof(double));
	cudaMalloc((void**) &dev_res, N * sizeof(double));

	cudaMemset(dev_res, 0, N * sizeof(double));

	cudaMemcpy(dev_BM_A_p, BM_A, BM_A_size* sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy(dev_x, x, N * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);

	// STEP 2: Invoke kernel to compute res
	cu_BM_V_MULT<<<GRID_SIZE, BLOCK_SIZE>>>(dev_BM_A_p, dev_x, dev_res, N, d);
	cudaDeviceSynchronize();

	// STEP 3: Copy data back to host
	cudaMemcpy(res, dev_res, N * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost);

	// STEP 4: Free Device memory and host memory
	cudaFree(dev_BM_A_p);
	cudaFree(dev_x);
	cudaFree(dev_res);
}

// construct N x (m-1) matrix h = [h1^T, h2^T, ..., hN^T]
Eigen::MatrixXd compute_H(const vector<double>& BM_A, int N, int d, int l, const Eigen::VectorXd & r){
	double *dev_BM_A_p, *dev_r, *dev_H, *H;
	int H_size = N * (2 * d - 1 - 1);
	
	// STEP 1: Allocate device memory and host memory
	cudaMalloc((void**) &dev_BM_A_p, BM_A.size() * sizeof(double));
	cudaMalloc((void**) &dev_r,	N * sizeof(double));
	cudaMalloc((void**) &dev_H, H_size * sizeof(double));
	H = (double*) malloc (H_size * sizeof(double));	

	cudaMemset(dev_H, 0, H_size * sizeof(double));
	memset(H, 0, H_size * sizeof(double));	

	// 应该还需要将 dev_H 的数据设置为 0，但是实验结果发现 default=0
	cudaMemcpy(dev_BM_A_p, BM_A.data(), BM_A.size() * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy(dev_r, r.data(), N * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);

	// STEP 2: Invoke kernel to compute H
	cu_compute_H<<<GRID_SIZE, BLOCK_SIZE>>>(dev_BM_A_p, dev_r, dev_H, N, d, l);
	cudaDeviceSynchronize();

	// STEP 3: Copy data back to host
	cudaMemcpy(H, dev_H, H_size * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost);
	Eigen::MatrixXd H_M = Eigen::Map<Eigen::MatrixXd>(H, 2 * d - 1 - 1, N).transpose();

	// STEP 4: Free Device memory and host memory
	cudaFree(dev_BM_A_p);
	cudaFree(dev_r);
	cudaFree(dev_H);
	free(H);

	return H_M;
}
void compute_H(double *H, double *BM_A, int BM_A_size, int N, int d, int l, double* r){
	double *dev_BM_A_p, *dev_r, *dev_H;
	int H_size = N * (2 * d - 1 - 1);
	
	// STEP 1: Allocate device memory and host memory
	cudaMalloc((void**) &dev_BM_A_p, BM_A_size * sizeof(double));
	cudaMalloc((void**) &dev_r,	N * sizeof(double));
	cudaMalloc((void**) &dev_H, H_size * sizeof(double));  

	cudaMemset(dev_H, 0, H_size * sizeof(double));

	// 应该还需要将 dev_H 的数据设置为 0，但是实验结果发现 default=0
	cudaMemcpy(dev_BM_A_p, BM_A, BM_A_size * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy(dev_r, r, N * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);

	// STEP 2: Invoke kernel to compute H
	cu_compute_H<<<GRID_SIZE, BLOCK_SIZE>>>(dev_BM_A_p, dev_r, dev_H, N, d, l);
	cudaDeviceSynchronize();

	// STEP 3: Copy data back to host
	cudaMemcpy(H, dev_H, H_size * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost);

	// STEP 4: Free Device memory and host memory
	cudaFree(dev_BM_A_p);
	cudaFree(dev_r);
	cudaFree(dev_H);

	return;
}

void copmute_STEP2(const vector<double>& BM_A, const Eigen::VectorXd& r,  const Eigen::MatrixXd& H, Eigen::VectorXd& u, Eigen::VectorXd& z, Eigen::MatrixXd& y, Eigen::MatrixXd& q, int N, int t, int d, int l){
	double *dev_BM_A_p, *dev_r, *dev_H, *dev_u, *dev_z, *dev_y, *dev_q;
	double *u_p, *z_p, *y_p, *q_p;
	int H_size = N * (2 * d - 1 - 1);
	int y_q_size = t * (2 * d - 1 -1);

	// STEP 1: Allocate device memory and host memory
	cudaMalloc((void**) &dev_BM_A_p, BM_A.size() * sizeof(double));
	cudaMalloc((void**) &dev_r,	N * sizeof(double));
	cudaMalloc((void**) &dev_H, H_size * sizeof(double));
	cudaMalloc((void**) &dev_u, t * sizeof(double));
	cudaMalloc((void**) &dev_z, t * sizeof(double));
	cudaMalloc((void**) &dev_y, y_q_size * sizeof(double));
	cudaMalloc((void**) &dev_q, y_q_size * sizeof(double));
	u_p = (double*) malloc (t * sizeof(double));
	z_p = (double*) malloc (t * sizeof(double));
	y_p = (double*) malloc (y_q_size * sizeof(double));
	q_p = (double*) malloc (y_q_size * sizeof(double));

	cudaMemset(dev_u, 0, t * sizeof(double));
	cudaMemset(dev_z, 0, t * sizeof(double));
	cudaMemset(dev_y, 0, y_q_size * sizeof(double));
	cudaMemset(dev_q, 0, y_q_size * sizeof(double));

	cudaMemcpy(dev_BM_A_p, BM_A.data(), BM_A.size() * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy(dev_r, r.data(), N * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);
	// 注意，不能使用 h.transpose().data(), 其 = h.data()
	cudaMemcpy(dev_H, H.data(), H_size * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);
	
	// STEP 2: Inoke kernel to compute u, z, y, q
	cu_compute_STEP2<<<GRID_SIZE, BLOCK_SIZE>>>(dev_BM_A_p, dev_r, dev_H, dev_u, dev_z, dev_y, dev_q, N, t, d, l);
	cudaDeviceSynchronize();

	// STEP 3: Copy data back to host
	cudaMemcpy(u_p, dev_u, t * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost);
	cudaMemcpy(z_p, dev_z, t * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost);
	cudaMemcpy(y_p, dev_y, y_q_size * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost);
	cudaMemcpy(q_p, dev_q, y_q_size * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost);
	u = Eigen::Map<Eigen::VectorXd>(u_p, t);
	z = Eigen::Map<Eigen::VectorXd>(z_p, t);
	y = Eigen::Map<Eigen::MatrixXd>(y_p, 2 * d - 1 - 1, t).transpose();
	q = Eigen::Map<Eigen::MatrixXd>(q_p, 2 * d - 1 - 1, t).transpose();

	// STEP 4: Free Device memory and host memory	
	cudaFree(dev_BM_A_p);
	cudaFree(dev_r);
	cudaFree(dev_H);
	cudaFree(dev_u);
	cudaFree(dev_z);
	cudaFree(dev_y);
	cudaFree(dev_q);
	free(u_p);
	free(z_p);
	free(y_p);
	free(q_p);
	return;
}
void copmute_STEP2(double *BM_A, int BM_A_size, double *r,  double* H, double* u, double* z, double* y, double* q, int N, int t, int d, int l){
	double *dev_BM_A, *dev_r, *dev_H, *dev_u, *dev_z, *dev_y, *dev_q;
	int H_size = N * (2 * d - 1 - 1);
	int y_q_size = t * (2 * d - 1 -1);

	// STEP 1: Allocate device memory and host memory
	cudaMalloc((void**) &dev_BM_A, BM_A_size * sizeof(double));
	cudaMalloc((void**) &dev_r,	N * sizeof(double));
	cudaMalloc((void**) &dev_H, H_size * sizeof(double));
	cudaMalloc((void**) &dev_u, t * sizeof(double));
	cudaMalloc((void**) &dev_z, t * sizeof(double));
	cudaMalloc((void**) &dev_y, y_q_size * sizeof(double));
	cudaMalloc((void**) &dev_q, y_q_size * sizeof(double));

	cudaMemset(dev_BM_A, 0, BM_A_size * sizeof(double));
	cudaMemset(dev_r, 0, N * sizeof(double));
	cudaMemset(dev_H, 0, H_size * sizeof(double));
	cudaMemset(dev_u, 0, t * sizeof(double));
	cudaMemset(dev_z, 0, t * sizeof(double));
	cudaMemset(dev_y, 0, y_q_size * sizeof(double));
	cudaMemset(dev_q, 0, y_q_size * sizeof(double));

	cudaMemcpy(dev_BM_A, BM_A, BM_A_size * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy(dev_r, r, N * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy(dev_H, H, H_size * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);
	
	// STEP 2: Inoke kernel to compute u, z, y, q
	cu_compute_STEP2<<<GRID_SIZE, BLOCK_SIZE>>>(dev_BM_A, dev_r, dev_H, dev_u, dev_z, dev_y, dev_q, N, t, d, l);
	cudaDeviceSynchronize();

	// STEP 3: Copy data back to host
	cudaMemcpy(u, dev_u, t * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost);
	cudaMemcpy(z, dev_z, t * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost);
	cudaMemcpy(y, dev_y, y_q_size * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost);
	cudaMemcpy(q, dev_q, y_q_size * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost);

	// STEP 4: Free Device memory and host memory	
	cudaFree(dev_BM_A);
	cudaFree(dev_r);
	cudaFree(dev_H);
	cudaFree(dev_u);
	cudaFree(dev_z);
	cudaFree(dev_y);
	cudaFree(dev_q);
	return;
}

Eigen::MatrixXd MM_LElem_MComp(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B){
	int m = A.rows(), n = A.cols(), k = B.cols();
	double* dev_A, *dev_B, *dev_R, *R;
	// STEP 1: Allocate device memory 
	cudaMalloc((void**) &dev_A, m * n * sizeof(double));
	cudaMalloc((void**) &dev_B, n * k * sizeof(double));
	cudaMalloc((void**) &dev_R, m * k * sizeof(double));
	R = (double*) malloc (m * k * sizeof(double));

	cudaMemset(dev_R, 0, m * k * sizeof(double));

	Eigen::MatrixXd AT = A.transpose();
	cudaMemcpy(dev_A, AT.data(), m * n * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy(dev_B, B.data(), n * k * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);

	// STEP 2: Invoke kernel to compute MM
	// Attention: result 的 size 非常小，每个元素的计算是一次内积的计算，所以
	//	考虑在每个元素的内部调用 dot 求解内积, 于是整体复杂度降至 lgN
	int block_size = 32;
	int grid_size = (m * k + block_size - 1) / block_size;
	cu_MM_LElem_MComp<<<grid_size, block_size>>>(dev_A, dev_B, dev_R, m, n, k);
	cudaDeviceSynchronize();

	// STEP 3: Copy data back to host
	cudaMemcpy(R, dev_R, m * k * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost);
	Eigen::MatrixXd R_mtx = Eigen::Map<Eigen::MatrixXd>(R, k, m).transpose();

	// STEP 4: Free device memory
	cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(dev_R);
	free(R);

	return R_mtx;
}
Eigen::MatrixXd MM_LElem_MComp(const Eigen::MatrixXd& A, Eigen::MatrixXd&& B){
	int m = A.rows(), n = A.cols(), k = B.cols();
	double* dev_A, *dev_B, *dev_R, *R;
	// STEP 1: Allocate device memory 
	cudaMalloc((void**) &dev_A, m * n * sizeof(double));
	cudaMalloc((void**) &dev_B, n * k * sizeof(double));
	cudaMalloc((void**) &dev_R, m * k * sizeof(double));
	R = (double*) malloc (m * k * sizeof(double));

	cudaMemset(dev_R, 0, m * k * sizeof(double));

	Eigen::MatrixXd AT = A.transpose();
	cudaMemcpy(dev_A, AT.data(), m * n * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy(dev_B, B.data(), n * k * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);

	// STEP 2: Invoke kernel to compute MM
	// Attention: result 的 size 非常小，每个元素的计算是一次内积的计算，所以
	//	考虑在每个元素的内部调用 dot 求解内积, 于是整体复杂度降至 lgN
	int block_size = 32;
	int grid_size = (m * k + block_size - 1) / block_size;
	cu_MM_LElem_MComp<<<grid_size, block_size>>>(dev_A, dev_B, dev_R, m, n, k);
	cudaDeviceSynchronize();

	// STEP 3: Copy data back to host
	cudaMemcpy(R, dev_R, m * k * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost);
	Eigen::MatrixXd R_mtx = Eigen::Map<Eigen::MatrixXd>(R, k, m).transpose();

	// STEP 4: Free device memory
	cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(dev_R);
	free(R);

	return R_mtx;
}
void MM_LElem_MComp(double *AT, double *B, double *R, int m, int n, int k){
	double* dev_AT, *dev_B, *dev_R;
	// STEP 1: Allocate device memory 
	cudaMalloc((void**) &dev_AT, m * n * sizeof(double));
	cudaMalloc((void**) &dev_B, n * k * sizeof(double));
	cudaMalloc((void**) &dev_R, m * k * sizeof(double));

	cudaMemset(dev_R, 0, m * k * sizeof(double));

	cudaMemcpy(dev_AT, AT, m * n * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy(dev_B, B, n * k * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);

	// STEP 2: Invoke kernel to compute MM
	// Attention: result 的 size 非常小，每个元素的计算是一次内积的计算，所以
	//	考虑在每个元素的内部调用 dot 求解内积, 于是整体复杂度降至 lgN
	int block_size = 32;
	int grid_size = (m * k + block_size - 1) / block_size;
	cu_MM_LElem_MComp<<<grid_size, block_size>>>(dev_AT, dev_B, dev_R, m, n, k);
	cudaDeviceSynchronize();

	// STEP 3: Copy data back to host
	cudaMemcpy(R, dev_R, m * k * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost);
	Eigen::MatrixXd R_mtx = Eigen::Map<Eigen::MatrixXd>(R, k, m).transpose();

	// STEP 4: Free device memory
	cudaFree(dev_AT);
	cudaFree(dev_B);
	cudaFree(dev_R);

	return;
}

Eigen::MatrixXd MM_MElem_LComp(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B){
	int m = A.rows(), n = A.cols(), k = B.cols();
	double* dev_A, *dev_B, *dev_R, *R;
	// STEP 1: Allocate device memory 
	cudaMalloc((void**) &dev_A, m * n * sizeof(double));
	cudaMalloc((void**) &dev_B, n * k * sizeof(double));
	cudaMalloc((void**) &dev_R, m * k * sizeof(double));
	R = (double*) malloc (m * k * sizeof(double));

	cudaMemset(dev_R, 0, m * k * sizeof(double));

	// 
	Eigen::MatrixXd AT = A.transpose();
	cudaMemcpy(dev_A, AT.data(), m * n * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy(dev_B, B.data(), n * k * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);

	// STEP 2: Invoke kernel to compute MM
	// Attention: result 的 size 非常小，每个元素的计算是一次内积的计算，所以
	//	考虑在每个元素的内部调用 dot 求解内积, 于是整体复杂度降至 lgN
	cu_MM_MElem_LComp<<<GRID_SIZE, BLOCK_SIZE>>>(dev_A, dev_B, dev_R, m, n, k);
	cudaDeviceSynchronize();

	// STEP 3: Copy data back to host
	cudaMemcpy(R, dev_R, m * k * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost);
	Eigen::MatrixXd R_mtx = Eigen::Map<Eigen::MatrixXd>(R, k, m).transpose();

	// STEP 4: Free device memory
	cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(dev_R);
	free(R);

	return R_mtx;
}
Eigen::MatrixXd MM_MElem_LComp(const Eigen::MatrixXd& A, Eigen::MatrixXd&& B){
	int m = A.rows(), n = A.cols(), k = B.cols();
	double* dev_A, *dev_B, *dev_R, *R;
	// STEP 1: Allocate device memory 
	cudaMalloc((void**) &dev_A, m * n * sizeof(double));
	cudaMalloc((void**) &dev_B, n * k * sizeof(double));
	cudaMalloc((void**) &dev_R, m * k * sizeof(double));
	R = (double*) malloc (m * k * sizeof(double));

	cudaMemset(dev_R, 0, m * k * sizeof(double));

	Eigen::MatrixXd AT = A.transpose();
	cudaMemcpy(dev_A, AT.data(), m * n * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy(dev_B, B.data(), n * k * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);

	// STEP 2: Invoke kernel to compute MM
	// Attention: result 的 size 非常小，每个元素的计算是一次内积的计算，所以
	//	考虑在每个元素的内部调用 dot 求解内积, 于是整体复杂度降至 lgN
	cu_MM_MElem_LComp<<<GRID_SIZE, BLOCK_SIZE>>>(dev_A, dev_B, dev_R, m, n, k);
	cudaDeviceSynchronize();

	// STEP 3: Copy data back to host
	cudaMemcpy(R, dev_R, m * k * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost);
	Eigen::MatrixXd R_mtx = Eigen::Map<Eigen::MatrixXd>(R, k, m).transpose();

	// STEP 4: Free device memory
	cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(dev_R);
	free(R);

	return R_mtx;
}
void MM_MElem_LComp(double *AT, double *B, double *R, int m, int n, int k){
	double* dev_A, *dev_B, *dev_R;
	// STEP 1: Allocate device memory 
	cudaMalloc((void**) &dev_A, m * n * sizeof(double));
	cudaMalloc((void**) &dev_B, n * k * sizeof(double));
	cudaMalloc((void**) &dev_R, m * k * sizeof(double));

	cudaMemset(dev_R, 0, m * k * sizeof(double));

	cudaMemcpy(dev_A, AT, m * n * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy(dev_B, B, n * k * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);

	// STEP 2: Invoke kernel to compute MM
	// Attention: result 的 size 非常小，每个元素的计算是一次内积的计算，所以
	//	考虑在每个元素的内部调用 dot 求解内积, 于是整体复杂度降至 lgN
	cu_MM_MElem_LComp<<<GRID_SIZE, BLOCK_SIZE>>>(dev_A, dev_B, dev_R, m, n, k);
	cudaDeviceSynchronize();

	// STEP 3: Copy data back to host
	cudaMemcpy(R, dev_R, m * k * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost);

	// STEP 4: Free device memory
	cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(dev_R);

	return;
}

Eigen::VectorXd compute_e1(const Eigen::MatrixXd& H, const Eigen::VectorXd& r, int N, int m){
	double *dev_H_i, *dev_r, *dev_e1_i, *e1;
	int H_size = N * (m - 1);
	// STEP 1: Allocate device memory
	cudaMalloc((void**) &dev_H_i,	N * sizeof(double));	// extract each i-th element in each h
	cudaMalloc((void**) &dev_r, 	N * sizeof(double));
	cudaMalloc((void**) &dev_e1_i, 	sizeof(double));
	e1 = (double*) malloc ( (m-1) * sizeof(double));

	cudaMemset(dev_e1_i, 0, sizeof(double));

	cudaMemcpy(dev_r, r.data(), N * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);

	// STEP 2: Invoke kernel to reduce
	for(int idx = 1; idx <= m - 1; idx++){
		Eigen::VectorXd H_col_i = H.col(idx - 1);
		cudaMemcpy(dev_H_i, H.data() + (idx - 1) * N, N * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);
		cudaMemset(dev_e1_i, 0, sizeof(double));
		cu_compute_e1_i<<<GRID_SIZE, BLOCK_SIZE>>>(dev_H_i, dev_r, dev_e1_i, N, m);
		cudaDeviceSynchronize();

		// STEP 3: Copy data back to host
		cudaMemcpy(&e1[idx - 1], dev_e1_i, sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost);
	}

	Eigen::VectorXd e1_vec = Eigen::Map<VectorXd>(e1, m-1);

	// STEP 4: Free device memory
	cudaFree(dev_H_i);
	cudaFree(dev_r);
	cudaFree(dev_e1_i);
	free(e1);

	return e1_vec;
}
void compute_e1(double *H, double *r, double *e1, int N, int m){
	double *dev_H_i, *dev_r, *dev_e1_i;
	int H_size = N * (m - 1);
	// STEP 1: Allocate device memory
	cudaMalloc((void**) &dev_H_i,	N * sizeof(double));	// extract each i-th element in each h
	cudaMalloc((void**) &dev_r, 	N * sizeof(double));
	cudaMalloc((void**) &dev_e1_i, 	sizeof(double));

	cudaMemset(dev_e1_i, 0, sizeof(double));

	cudaMemcpy(dev_r, r, N * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);

	// STEP 2: Invoke kernel to reduce
	for(int idx = 1; idx <= m - 1; idx++){
		cudaMemcpy(dev_H_i, H + (idx - 1) * N, N * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);
		cudaMemset(dev_e1_i, 0, sizeof(double));
		cu_compute_e1_i<<<GRID_SIZE, BLOCK_SIZE>>>(dev_H_i, dev_r, dev_e1_i, N, m);
		cudaDeviceSynchronize();

		// STEP 3: Copy data back to host
		cudaMemcpy(&e1[idx - 1], dev_e1_i, sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost);
	}

	// Eigen::VectorXd e1_vec = Eigen::Map<VectorXd>(e1, m-1);

	// STEP 4: Free device memory
	cudaFree(dev_H_i);
	cudaFree(dev_r);
	cudaFree(dev_e1_i);

	return;
}

Eigen::VectorXd compute_rbM_inverse_vec(const Eigen::VectorXd& z, const Eigen::VectorXd& r, int N, int m, int t, int l){
	double *dev_z, *dev_r, *dev_res, *res;

	// STEP 1: Allocate memory
	cudaMalloc((void**) &dev_z, t * sizeof(double));
	cudaMalloc((void**) &dev_r, N * sizeof(double));
	cudaMalloc((void**) &dev_res, t * sizeof(double));
	res = (double*) malloc (t * sizeof(double));

	cudaMemset(dev_res, 0, t * sizeof(double));

	cudaMemcpy(dev_z, z.data(), t * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy(dev_r, r.data(), N * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);

	// STEP 2: Invoke kernel
	cu_copmute_rbM_inverse_vec<<<GRID_SIZE, BLOCK_SIZE>>>(dev_z, dev_r, dev_res, N, m, t, l);
	cudaDeviceSynchronize();

	// STEP 3: Copy data back to host
	cudaMemcpy(res, dev_res, t * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost);
	Eigen::VectorXd rbm_inverse_vec = Eigen::Map<Eigen::VectorXd>(res, t);

	// STEP 4: Free Memory
	cudaFree(dev_z);
	cudaFree(dev_r);
	cudaFree(dev_res);
	free(res);

	return rbm_inverse_vec;
}
void compute_rbM_inverse_vec(double *z, double *r, double *res, int N, int m, int t, int l){
	double *dev_z, *dev_r, *dev_res;

	// STEP 1: Allocate memory
	cudaMalloc((void**) &dev_z, t * sizeof(double));
	cudaMalloc((void**) &dev_r, N * sizeof(double));
	cudaMalloc((void**) &dev_res, t * sizeof(double));

	cudaMemset(dev_res, 0, t * sizeof(double));

	cudaMemcpy(dev_z, z, t * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy(dev_r, r, N * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);

	// STEP 2: Invoke kernel
	cu_copmute_rbM_inverse_vec<<<GRID_SIZE, BLOCK_SIZE>>>(dev_z, dev_r, dev_res, N, m, t, l);
	cudaDeviceSynchronize();

	// STEP 3: Copy data back to host
	cudaMemcpy(res, dev_res, t * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost);

	// STEP 4: Free Memory
	cudaFree(dev_z);
	cudaFree(dev_r);
	cudaFree(dev_res);

	return;
}

Eigen::MatrixXd RowTrans(const Eigen::VectorXd& coeff_vec, const Eigen::MatrixXd& mtx){
	int m = mtx.rows(), n = mtx.cols();
	double *dev_coeff_vec, *dev_mtx, *dev_res, *res;
	// STEP 1: Allocate memory
	cudaMalloc((void**) &dev_coeff_vec, m * sizeof(double));
	cudaMalloc((void**) &dev_mtx, m * n * sizeof(double));
	cudaMalloc((void**) &dev_res, m * n * sizeof(double));
	res = (double*) malloc (m * n * sizeof(double));

	cudaMemset(dev_res, 0, m * n * sizeof(double));

	cudaMemcpy(dev_coeff_vec, coeff_vec.data(), m * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy(dev_mtx, mtx.data(), m * n * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);

	// STEP 2: Invoke Kernel
	cu_RowTrans<<<GRID_SIZE, BLOCK_SIZE>>>(dev_coeff_vec, dev_mtx, dev_res, m, n);
	cudaDeviceSynchronize();

	// STEP 3: Copy back
	cudaMemcpy(res, dev_res, m * n * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost);
	Eigen::MatrixXd res_mtx = Eigen::Map<Eigen::MatrixXd>(res, m, n);
	
	// STEP 4: Free memory
	cudaFree(dev_coeff_vec);
	cudaFree(dev_mtx);
	cudaFree(dev_res);
	free(res);

	return res_mtx;
} 
void RowTrans(double* coeff_vec, double* mtx, double* res, int m, int n){
	double *dev_coeff_vec, *dev_mtx, *dev_res;
	// STEP 1: Allocate memory
	cudaMalloc((void**) &dev_coeff_vec, m * sizeof(double));
	cudaMalloc((void**) &dev_mtx, m * n * sizeof(double));
	cudaMalloc((void**) &dev_res, m * n * sizeof(double));

	cudaMemset(dev_res, 0, m * n * sizeof(double));

	cudaMemcpy(dev_coeff_vec, coeff_vec, m * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy(dev_mtx, mtx, m * n * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);

	// STEP 2: Invoke Kernel
	cu_RowTrans<<<GRID_SIZE, BLOCK_SIZE>>>(dev_coeff_vec, dev_mtx, dev_res, m, n);
	cudaDeviceSynchronize();

	// STEP 3: Copy back
	cudaMemcpy(res, dev_res, m * n * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost);
	
	// STEP 4: Free memory
	cudaFree(dev_coeff_vec);
	cudaFree(dev_mtx);
	cudaFree(dev_res);

	return;
}

Eigen::VectorXd RestoreD(const Eigen::VectorXd& s, const Eigen::VectorXd& alpha, int N, int d, int l, int m, int t){
	double *dev_s, *dev_alpha, *dev_D, *D;
	// STEP 1: Allocate Device Memory
	cudaMalloc((void**) &dev_s, (m - 1) * sizeof(double));
	cudaMalloc((void**) &dev_alpha, t * sizeof(double));
	cudaMalloc((void**) &dev_D, N * sizeof(double));

	D = (double*) malloc (N * sizeof(double));
	cudaMemset(dev_D, 0, N * sizeof(double));

	cudaMemcpy(dev_s, s.data(), (m - 1) * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy(dev_alpha, alpha.data(), t * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);

	// STEP 2: invoke kernel
	cu_RestoreD<<<GRID_SIZE, BLOCK_SIZE>>>(dev_s, dev_alpha, dev_D, N, d, l, m, t);
	cudaDeviceSynchronize();

	// STEP 3: Copy data back
	cudaMemcpy(D, dev_D, N * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost);
	Eigen::VectorXd D_vec = Eigen::Map<Eigen::VectorXd>(D, N);

	// STEP 4: Free Memory
	cudaFree(dev_s);
	cudaFree(dev_alpha);
	cudaFree(dev_D);
	free(D);

	return D_vec;
}

void RestoreD(double *s, double *alpha, double *D, int N, int d, int l, int m, int t){
	double *dev_s, *dev_alpha, *dev_D;
	// STEP 1: Allocate Device Memory
	cudaMalloc((void**) &dev_s, (m - 1) * sizeof(double));
	cudaMalloc((void**) &dev_alpha, t * sizeof(double));
	cudaMalloc((void**) &dev_D, N * sizeof(double));

	cudaMemset(dev_D, 0, N * sizeof(double));

	cudaMemcpy(dev_s, s, (m - 1) * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy(dev_alpha, alpha, t * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);

	// STEP 2: invoke kernel
	cu_RestoreD<<<GRID_SIZE, BLOCK_SIZE>>>(dev_s, dev_alpha, dev_D, N, d, l, m, t);
	cudaDeviceSynchronize();

	// STEP 3: Copy data back
	cudaMemcpy(D, dev_D, N * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost);

	// STEP 4: Free Memory
	cudaFree(dev_s);
	cudaFree(dev_alpha);
	cudaFree(dev_D);

	return;
}

void Trans(double *src, double *dst, int m, int n){
	double *dev_src, *dev_dest;
	// STEP 1: Allocate device memory
	cudaMalloc((void**) &dev_src, m * n * sizeof(double));
	cudaMalloc((void**) &dev_dest, m * n * sizeof(double));

	cudaMemset(dev_dest, 0, m * n * sizeof(double));

	cudaMemcpy(dev_src, src, m * n * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);

	// STEP 2: Invoke kernel 
	cu_GetTranspose<<<GRID_SIZE, BLOCK_SIZE>>>(dev_src, dev_dest, m, n);
	cudaDeviceSynchronize();

	// STEP 3: Copy data back
	cudaMemcpy(dst, dev_dest, m * n * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost);

	// STEP 4: Free
	cudaFree(dev_src);
	cudaFree(dev_dest);
	return;
}

void ADD(double* vec1, double* vec2, double *res, int n){
	double *dev_vec1, *dev_vec2, *dev_res;	
	// STEP 1: Allocate Device Memory
	cudaMalloc((void**) &dev_vec1, n * sizeof(double));
	cudaMalloc((void**) &dev_vec2, n * sizeof(double));
	cudaMalloc((void**) &dev_res, n * sizeof(double));

	cudaMemset(dev_res, 0, n * sizeof(double));
	
	cudaMemcpy(dev_vec1, vec1, n * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy(dev_vec2, vec2, n * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);

	// STEP 2: Invoke Kernel
	cu_Add<<<GRID_SIZE, BLOCK_SIZE>>>(dev_vec1, dev_vec2, dev_res, n);
	cudaDeviceSynchronize();

	// STEP 3: Copy data back to host
	cudaMemcpy(res, dev_res, n * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost);

	// STEP 4: Free
	cudaFree(dev_vec1);
	cudaFree(dev_vec2);
	cudaFree(dev_res);

	return;
}

void SUB(double* vec1, double* vec2, double *res, int n){
	double *dev_vec1, *dev_vec2, *dev_res;	
	// STEP 1: Allocate Device Memory
	cudaMalloc((void**) &dev_vec1, n * sizeof(double));
	cudaMalloc((void**) &dev_vec2, n * sizeof(double));
	cudaMalloc((void**) &dev_res, n * sizeof(double));

	cudaMemset(dev_res, 0, n * sizeof(double));
	
	cudaMemcpy(dev_vec1, vec1, n * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy(dev_vec2, vec2, n * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);

	// STEP 2: Invoke Kernel
	cu_Sub<<<GRID_SIZE, BLOCK_SIZE>>>(dev_vec1, dev_vec2, dev_res, n);
	cudaDeviceSynchronize();

	// STEP 3: Copy data back to host
	cudaMemcpy(res, dev_res, n * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost);

	// STEP 4: Free
	cudaFree(dev_vec1);
	cudaFree(dev_vec2);
	cudaFree(dev_res);

	return;
}

void Scalar(double* vec1, double *res, double scalar, int n){
	double *dev_vec1, *dev_res;	
	// STEP 1: Allocate Device Memory
	cudaMalloc((void**) &dev_vec1, n * sizeof(double));
	cudaMalloc((void**) &dev_res, n * sizeof(double));

	cudaMemset(dev_res, 0, n * sizeof(double));
	
	cudaMemcpy(dev_vec1, vec1, n * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);

	// STEP 2: Invoke Kernel
	cu_Scalar<<<GRID_SIZE, BLOCK_SIZE>>>(dev_vec1, dev_res, scalar, n);
	cudaDeviceSynchronize();

	// STEP 3: Copy data back to host
	cudaMemcpy(res, dev_res, n * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost);

	// STEP 4: Free
	cudaFree(dev_vec1);
	cudaFree(dev_res);

	return;
}