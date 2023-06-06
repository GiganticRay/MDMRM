#ifndef _SOLVER_H_
#define _SOLVER_H_
#include "CommonHeader.h"
#include "Helper.h"
#include "CUDAKernel.cuh"

enum METHOD{
	MR 				= 1,
	ROWADJUSTMR		= 2,
	COLADJUSTDIAGMR	= 3,
	CGN				= 4,
	BI_CG			= 5,
	PARALLEL_SPIKE	= 6,
	MDMRB			= 7,
	JACOBI			= 8,
	PARALLEL_MR		= 9,
	PARALLEL_MDMRB	= 10,
	PARALLEL_MDMRB_WITHOUT_EIGEN = 11,
	PARALLEL_CGN	= 12,
	PARALLEL_BICG	= 13,
	PARALLEL_CG		= 14
};

class LinearSystem{
public:
	void SetBM_AFromEigen(vector<double>& BM_A, const SpMat& A, int d, int N);
	
public:
	SpMat A;
	// vector<vector<double>> BM_A;
	// 采取 https://netlib.org/lapack/lug/node124.html 的压缩格式压缩 banded matrix A, 对 BM_A 的操作都要做特殊处理，写作 BM_(OP), 比如 BM_V_MULT, 即是 BM 与 vector 的乘法
	vector<double> BM_A;
	Eigen::VectorXd x;
	Eigen::VectorXd b;
	int N;
	int m;
	int d;

	int max_step;
	double res_THOLD;
	double res_diff_THOLD;

	LinearSystem(){}

	LinearSystem(const Eigen::MatrixXd A, const Eigen::VectorXd b, int d);

	LinearSystem(string file_path, int d, int max_step = 1000, double res_THOLD = 1e-20, double res_diff_THOLD = 0);
};

class IterativeSolver{
private:
	LinearSystem ls;
	
public:
	IterativeSolver(string file_path, int d, int max_step = 1000, double res_THOLD = 1e-02, double res_diff_THOLD = 0);

	IterativeSolver(Eigen::MatrixXd A, Eigen::VectorXd b, int d);

	Eigen::VectorXd mult(Eigen::MatrixXd A, int m, Eigen::VectorXd x);

	/* MR Iteration
		r = b - Ax
		while (not convergence):
			\alpha = (Ar, r) / (Ar, Ar)
			x = x + \alpha r
			r = r - \alpha A r
	*/
	void MR();

	/* row adjust MR Iteration
		r = b - Ax
		while(not convergence):
			for i in range(N):
				alpha_i = -r_i / 2(A[i, *], r)
		// 这样算不出来 x 的迭代表达式，但是我可以看一下， residual 是否是逐渐减小的: 是的
	*/
	void RowAdjustMR();

	/*
		algorithm for diagonal matrix verification (col adjust MR Iteration)
		r = b - Ax
		while(not convergence):
			for i in range(N):
				alpha_i = 1 / aii (就相当于是逆了，一次求解)
	*/
	void ColAdjustDiagMR();

	// Conjugate gradients on the normal equations
	void CGN();

	void BI_CG();

	Eigen::VectorXd Parallel_Spike(int recursive_level);

	void MDMRB();

	void JACOBI();

	void PARALLEL_MR();

	void PARALLEL_MDMRB();

	void PARALLEL_MDMRB_WITHOUT_EIGEN();

	void PARALLEL_CGN();

	void PARALLEL_BICG();

	void PARALLEL_CG();
};

#endif