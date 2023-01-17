#ifndef _CUDA_KERNEL_CUH_
#define _CUDA_KERNEL_CUH_

#include "CommonHeader.h"
#include "cuda_runtime.h"

inline cudaError_t checkCuda(cudaError_t result){
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
	return result;
}

// wrapper
double dot(const Eigen::VectorXd & v1, const Eigen::VectorXd & v2);
double dot(double *v1, double *v2, int N);

/**
 * BM_A is banded matrix stored in a LAPACK formation.
 * parallel the for loop to compute final res
*/
Eigen::VectorXd BM_V_MULT(vector<double> BM_A, int N, int d, Eigen::VectorXd x);
void BM_V_MULT(double *BM_A, int BM_A_size, double *x, double *res, int N, int d);

Eigen::MatrixXd compute_H(const vector<double>& BM_A, int N, int d, int l, const Eigen::VectorXd & r);
void compute_H(double *H, double *BM_A, int BM_A_size, int N, int d, int l, double* r);

void copmute_STEP2(const vector<double>& BM_A, const Eigen::VectorXd& r, const Eigen::MatrixXd& H, Eigen::VectorXd& u, Eigen::VectorXd& z, Eigen::MatrixXd& y, Eigen::MatrixXd& q, int N, int t, int d, int l);
void copmute_STEP2(double *BM_A, int BM_A_size, double *r,  double* H, double* u, double* z, double* y, double* q, int N, int t, int d, int l);

// 专项优化用于求解 (m-1) * t * t * (m-1) 的情况，每个元素内部用内积搞定
// 并且由于 m 肯定不大，所以在外层并行中，分配的线程数量也要做优化
// MatrixMatrix Multiplication with the situation of Less elements - plenty of computation of each element
Eigen::MatrixXd MM_LElem_MComp(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B);
Eigen::MatrixXd MM_LElem_MComp(const Eigen::MatrixXd& A, Eigen::MatrixXd&& B);
void MM_LElem_MComp(double *AT, double *B, double *R, int m, int n, int k);

// 专项优化永于求解t * (m-1) * (m-1) * x 的情况，元素众多但是每个元素计算量很小
// MatrixMatrix Multiplication with the situation of plenty of elements - less computation of each element
Eigen::MatrixXd MM_MElem_LComp(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B);
Eigen::MatrixXd MM_MElem_LComp(const Eigen::MatrixXd& A, Eigen::MatrixXd&& B);
void MM_MElem_LComp(double *AT, double *B, double *R, int m, int n, int k);

// 注意：计算 e1 时不能够直接用 reduction ，因为 shared memory 必须使用常数初始化
Eigen::VectorXd compute_e1(const Eigen::MatrixXd& h, const Eigen::VectorXd& r, int N, int m);
void compute_e1(double *h, double *r, double *e1, int N, int m);

Eigen::VectorXd compute_rbM_inverse_vec(const Eigen::VectorXd& z, const Eigen::VectorXd& r, int N, int m, int t, int l);
void compute_rbM_inverse_vec(double *z, double *r, double *res, int N, int m, int t, int l);

Eigen::MatrixXd RowTrans(const Eigen::VectorXd& coeff_vec, const Eigen::MatrixXd& mtx);
// mtx has size (m x n)
void RowTrans(double* coeff_vec, double* mtx, double* res, int m, int n);

Eigen::VectorXd RestoreD(const Eigen::VectorXd& s, const Eigen::VectorXd& alpha, int N, int d, int l, int m, int t);
void RestoreD(double *s, double *alpha, double *D, int N, int d, int l, int m, int t);

void Trans(double *src, double *dst, int m, int n);

void ADD(double* vec1, double* vec2, double *res, int n);
void SUB(double* vec1, double* vec2, double *res, int n);

void Scalar(double* vec1, double *res, double scalar, int n);
#endif