#include "Helper.h"

void saveData(std::string fileName, Eigen::MatrixXi matrix)
{
    //https://eigen.tuxfamily.org/dox/structEigen_1_1IOFormat.html
    const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");
 
    std::ofstream file(fileName);
    if (file.is_open())
    {
        file << matrix.format(CSVFormat);
        file.close();
    }
}

// https://aleksandarhaber.com/eigen-matrix-library-c-tutorial-saving-and-loading-data-in-from-a-csv-file/
Eigen::MatrixXi openData(std::string fileToOpen)
{
    std::vector<int> matrixEntries;
    std::ifstream matrixDataFile(fileToOpen);
    std::string matrixRowString;
    std::string matrixEntry;
    int matrixRowNumber = 0;
 
    while (getline(matrixDataFile, matrixRowString)) 
    {
        stringstream matrixRowStringStream(matrixRowString); 
 
        while (getline(matrixRowStringStream, matrixEntry, ',')) 
        {
            matrixEntries.push_back(stod(matrixEntry));   
        }
        matrixRowNumber++; 
    }
 
    return Map<Matrix<int, Dynamic, Dynamic, RowMajor>>(matrixEntries.data(), matrixRowNumber, matrixEntries.size() / matrixRowNumber);
}

void tool_Generate_Random_vec(int length, double* b){
	if(b == NULL){
		exit(-1);
	}
	for(int n = 0; n < length; n++){
		b[n] = rand() % 10;
	}
}

// load matrix in matrix market form into CSR Eigen Form ( Matrix market => triplet array => CSR Eigen Form)
SpMat Init_SpMatirx(string filePath){
    FILE 	*f;
	MM_typecode matcode;
	int 	M, N, nz;
	int 	i, *I, *J;
	double 	*val;
	int 	ret_code;

	if ((f = fopen(filePath.c_str(), "r")) == NULL) {
        printf("open file error!\n");
        exit(1);
    }
	
	// determine the type of matrix being represented in a Matrix Market file
    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }

	// get the basic info of sparse matrix
	if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) !=0)
        exit(1);

    bool is_symmetry = mm_is_symmetric(matcode);

    vector<int> I_vec;
    vector<int> J_vec;
    vector<double> val_vec;
    int x, y;
    double v;

	for (i=0; i < nz; i++){
        fscanf(f, "%d %d %lg\n", &x, &y, &v);
        x--;  /* adjust from 1-based to 0-based */
        y--;
        I_vec.push_back(x);
        J_vec.push_back(y);
        val_vec.push_back(v);
        if(is_symmetry && (x != y)){
            I_vec.push_back(y);
            J_vec.push_back(x);
            val_vec.push_back(v);
        }
    }
    nz = I_vec.size();  // reset nz in the case of symmetry matrix

    if (f != stdin) fclose(f);

	// construct Eigen sparse matrix obj
	vector<T> triplet_vec;
	triplet_vec.reserve(nz);
	for(int i = 0; i < nz; i++)
		triplet_vec.push_back(T(I_vec[i], J_vec[i], val_vec[i]));
	
	SpMat A(M, N);
	A.setFromTriplets(triplet_vec.begin(), triplet_vec.end());
    return A;
}

// 判定 double 是否相等
bool equal_double(double l, double r){
    return fabs(l - r) <= __DBL_EPSILON__;
}

vector<int> get_OPT_Comb(const Eigen::VectorXd& A_r, int m){
    int n = A_r.size();

    vector<int> ans;
    int ans_l = -1;
    double max = __DBL_MIN__;

    // 看哪种收益最高
    for(int l = 0; l < m; l++){
        double curr_max = 0;
        for(int L = l; L < n; L += m){
            curr_max += fabs(A_r[L]);
        }
        if(max < curr_max){
            max = curr_max;
            ans_l = l;
        }
    }
    for(int L = ans_l; L < n; L += m){
        ans.push_back(L);
    }

    return ans;
}

vector<int> get_Rand_Comb(int n, int m){
    vector<int> ans;
    int rand_start = rand() % m;
    for(int l = rand_start; l < m; l += m){
        ans.push_back(l);
    }
    return ans;
}

// 将 A 的系数归一化
void Normalize(Eigen::SparseMatrix<double>& A){
    double max_item = __DBL_MIN__;
    for (int k = 0; k < A.outerSize(); ++k){
        for (SparseMatrix<double>::InnerIterator it(A, k); it; ++it){
            max_item = max(it.value(), max_item);
        }
    }
    for (int k = 0; k < A.outerSize(); ++k){
        for (SparseMatrix<double>::InnerIterator it(A, k); it; ++it){
            it.valueRef() = it.value() / (0.1 * max_item);
        }
    }
}

int lower(int i, int N, int d){
	return max(1, i - d + 1);
}

int upper(int i, int N, int d){
	return min(N, i + d - 1);
}

pair<int, int> ConvertCoordsFromDenseToLAPACK(int row_idx, int col_idx, int N, int d){
    int row_in_BM = d - 1 + 1 + row_idx - col_idx;
    return {row_in_BM, col_idx};
}

pair<int, int> ConvertCoordsFromLAPACKToDense(int row_in_BM, int col_in_BM, int N, int d){
    int row_in_Dense = row_in_BM - (d - 1) - 1 + col_in_BM;
    return {row_in_Dense, col_in_BM};
}