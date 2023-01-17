#ifndef __COMMONHEADER__
#define __COMMONHEADER__

#include <stdlib.h>
#include <stdio.h>
#include <iostream>

#include "CycleTimer.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Core>

#include <vector>
#include <math.h>
#include "omp.h"

#include <cmath>
#include <cfloat>

#include <assert.h>

#include <numeric>

using namespace std;
using namespace Eigen;

using Eigen::internal::BandMatrix;
typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<double> T;

#endif