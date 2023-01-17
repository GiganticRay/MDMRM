#include <stdio.h>
#include <mkl.h>
#include "spike.h"
#include "mmio.h"

#define M(p, j, i) p[j + ld##p * (i)]	// fortran style matrix macro

/**
 * filePath: 	matrix-market file
 * A:			pointer to the banded-storage matrix
 * n:			size of original coefficient matrix(square)
 * d:			half the non-zero diagonal.
 * 
 **/
void InitMatrixFromFile(char* filePath, double*** A, double***oA, double **b, double** ob, int n, int d){
	FILE 	*f;
	MM_typecode matcode;
	int 	M, N, nz;
	int 	i, j, *I, *J;
	double 	*val;
	int 	ret_code;

	if ((f = fopen(filePath, "r")) == NULL) {
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

	I = 	(int *) malloc(nz * sizeof(int));
    J = 	(int *) malloc(nz * sizeof(int));
    val = 	(double *) malloc(nz * sizeof(double));

	for (i=0; i < nz; i++){
        fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i]);
        I[i]--;  /* adjust from 1-based to 0-based */
        J[i]--;
    }
    if (f != stdin) fclose(f);

	int m = 2 * d - 1;
	if(*A == NULL){
		*A = (double**) malloc (sizeof(double*) * m);
		for(i = 0; i < m; i++){
			(*A)[i] = (double*) malloc (sizeof(double) * n);
		}
	}
	if(*oA == NULL){
		*oA = (double**) malloc (sizeof(double*) * m);
		for(i = 0; i < m; i++){
			(*oA)[i] = (double*) malloc (sizeof(double) * n);
		}
	}

	if(*b == NULL){
		*b = (double*) malloc (sizeof(double) * n);
	}
	if(*ob == NULL){
		*ob = (double*) malloc (sizeof(double) * n);
	}

	for(i = 0; i < nz; i++){
		int off_diags = J[i] - I[i];
		int x_in_A = d - off_diags - 1;
		int y_in_A = J[i]; 
		(*A)[x_in_A][y_in_A] = val[i];	
		(*oA)[x_in_A][y_in_A] = val[i];

		// Initiate b
		(*b)[I[i]] += (val[i] / 2);
		(*ob)[I[i]] += (val[i] / 2);
	}
};

int main(){
	int spikeparam[64];
	const char trans = 'N';

	int i, j;
	int n = 237, d = 5, m = 2*d-1, kl = 4, ku = 4, nrhs = 1, ldA = 2*d-1, ldf = n, ldoA = ldA, ldof = n, info, klu;
	klu = (kl >= ku) ? kl : ku;

	char* filePath = "/public/home/LeiChao/Document/MRForMDiag/data/nos1.mtx";
	double **A 	= NULL;
	double **oA = NULL;
	double *f	= NULL;
	double *of	= NULL;
	InitMatrixFromFile(filePath, &A, &oA, &f, &of, n, d);

	printf("construct A, b finished!\n");

	// initialize the spike params array
	spikeinit (spikeparam ,&n ,&klu);
	printf("initialize spike paras finished!\n");

	// transform A** to A*
	double* A_hat = (double *) malloc(ldA * n * sizeof(double));
	double* oA_hat = (double *) malloc(ldA * n * sizeof(double));
	int ldA_hat = ldA;
	int ldoA_hat = ldoA;
	for(i = 0; i < ldA; i++){
		for(j = 0; j < n; j++){
			M(A_hat, i, j) = A[i][j];
			M(oA_hat, i, j) = A[i][j];
		}
	}

	// Instruct SPIKE to print timing and partitioning info
	// Note that the index is off by one, because we're using C
	spikeparam[0] = 1;	// Print Flag
	spikeparam[19] = 6;		// number of partitions into which SPIKE is broken
	// spikeparam[20]= 1;	// number of recursive levels for reduced system
	// spikeparam[21]= 64;	// number of threads used by SPIKE

	dspike_gbsv(spikeparam ,&n ,&kl ,&ku ,&nrhs ,A_hat ,&ldA ,f ,&ldf ,&info);
	printf("gbsv finished!\n");

	double res , d_one = 1.0 , d_mone = -1.0;
	int i_one = 1;
	// Let's just check the residual of the first vector,
	// Since we only have one vector by default
	DGBMV (&trans , &n , &n , &kl , &ku ,
		&d_mone , oA_hat , &ldA , &M(f ,0 ,0) ,&i_one ,
		&d_one , &M(of, 0, 0) , &i_one);
		res = M(of, idamax(&n, &M(of, 0, 0), &i_one) - 1, 0);

	printf ("n, \t kl, \t ku, \t nrhs, \t residual \n");
	printf ("%d, \t %d, \t %d, \t %d, \t %.3E \n" ,n , kl , ku , nrhs , res );

	return 0; 
}