#include "CommonHeader.h"
#include "Solver.h"

int main(int argc, char* argv[]){
	// initialize, read parameters from command line
	srand((unsigned int)(time(NULL)));
	// load sparse matrix in CSR Form
	std::string filePath(argv[1]);
	// SpMat A = Init_SpMatirx(filePath);
	int d = stoi(string(argv[2]));
	int iterative_num = stoi(string(argv[3]));
	int FUNC = stoi(string(argv[4]));

	IterativeSolver ite_solv = IterativeSolver(filePath, d, iterative_num);

	switch (FUNC)
	{
	case MR:
		ite_solv.MR();
		break;
	case ROWADJUSTMR:
		ite_solv.RowAdjustMR();
		break;
	case COLADJUSTDIAGMR:
		ite_solv.ColAdjustDiagMR();
		break;
	case CGN:
		ite_solv.CGN();
		break;
	case BI_CG:
		ite_solv.BI_CG();
		break;
	case PARALLEL_SPIKE:
		ite_solv.Parallel_Spike(1);
		break;
	case MDMRM:
		ite_solv.MDMRM();
		break;
	case JACOBI:
		ite_solv.JACOBI();
		break;
	case PARALLEL_MR:
		ite_solv.PARALLEL_MR();
		break;
	case PARALLEL_MDMRM:
		ite_solv.PARALLEL_MDMRM();
		break;
	case PARALLEL_MDMRM_WITHOUT_EIGEN:
		ite_solv.PARALLEL_MDMRM_WITHOUT_EIGEN(); 
		break;
	case PARALLEL_CGN:
		ite_solv.PARALLEL_CGN();
		break;
	case PARALLEL_BICG:
		ite_solv.PARALLEL_BICG();
		break;
	case PARALLEL_CG:
		ite_solv.PARALLEL_CG();
		break;
	default:
		std::cout << "# Pls check the func you choose!" << endl;
		break;
	};
	return 0;
}
