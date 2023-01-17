#ifndef HELPER_H
#define HELPER_H

#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <fstream>
#include <string>
#include <queue>

#include "mmio.h"

using namespace std;
using namespace Eigen;

typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<double> T;
 
void saveData(std::string fileName, Eigen::MatrixXi matrix);

// https://aleksandarhaber.com/eigen-matrix-library-c-tutorial-saving-and-loading-data-in-from-a-csv-file/
Eigen::MatrixXi openData(std::string fileToOpen);

void tool_Generate_Random_vec(int length, double* b);

// load matrix in matrix market form into CSR Eigen Form ( Matrix market => triplet array => CSR Eigen Form)
SpMat Init_SpMatirx(string filePath);

// 判定 double 是否相等
bool equal_double(double l, double r);

vector<int> get_OPT_Comb(const Eigen::VectorXd& A_r, int m);

vector<int> get_Rand_Comb(int n, int m);

// 将 A 的系数归一化
void Normalize(Eigen::SparseMatrix<double>& A);

int lower(int i, int N, int d);
int upper(int i, int N, int d);

pair<int, int> ConvertCoordsFromDenseToLAPACK(int row_idx, int col_idx, int N, int d);

pair<int, int> ConvertCoordsFromLAPACKToDense(int row_in_BM, int col_in_BM, int N, int d);

#endif