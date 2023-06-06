#include "Solver.h"

/*
	Based on banded formation in https://netlib.org/lapack/lug/node124.html
	A[i][j] is stored in BM_A[ku + 1 + i - j][j] where kd = d-1.
	The index are 0-based.
*/
void LinearSystem::SetBM_AFromEigen(vector<double>& BM_A, const SpMat& A, int d, int N){
	BM_A.resize((2 * d - 1) * N, 0);
	for(int i = 0; i < N; i++){
		int row_idx		= i + 1;
		int low_col_idx	= lower(row_idx, N, d);
		int up_col_idx	= upper(row_idx, N, d);
		for(int col_idx = low_col_idx; col_idx <= up_col_idx; col_idx++){
			auto [row_in_BM, col_in_BM] = ConvertCoordsFromDenseToLAPACK(row_idx, col_idx, N, d);
			BM_A[(row_in_BM - 1) * N + col_in_BM - 1] = A.coeff(row_idx - 1, col_idx - 1);
		}
	}
	return;
}

LinearSystem::LinearSystem(const Eigen::MatrixXd A, const Eigen::VectorXd b, int d){
	this->d = d;
	this->A = A.sparseView();
	this->b = b;
	this->N = A.cols();
	this->m = 2 * d - 1;
	x = Eigen::VectorXd(N);
	x.setOnes();

	this->SetBM_AFromEigen(this->BM_A, this->A, this->d, this->N);
}

LinearSystem::LinearSystem(string file_path, int d, int max_step /*= 1000*/, double res_THOLD /*= 1e-02*/, double res_diff_THOLD /*= 0*/){
	this->max_step = max_step;
	this->res_THOLD = res_THOLD;
	this->res_diff_THOLD = res_diff_THOLD;

	// the data file should be csr format.
	double start_read_data_time = CycleTimer::currentSeconds();
	cout << "# Reading DATA..." << endl;
	A = Init_SpMatirx(file_path);
	cout << "# Reading Matrix Successfully!" << endl;

	N = A.cols();
	this->d = d;
	m = 2 * d - 1;
	b = Eigen::VectorXd(N);
	x = Eigen::VectorXd(N);
	x.setOnes();

	// initialize random B
	/*
	cout << "# Initialize b..." << endl;
	double* b_val_p = (double*) malloc (sizeof(double) * N);
	tool_Generate_Random_vec(N, b_val_p);

	// assign bias vector and initialize solution.
	for(int i = 0; i < N; i++){
		b[i] = b_val_p[i];
		x[i] = 1;
	}

	// 将 b 设置为 A 相关
	b = A * Eigen::VectorXd::Constant(N, 0.5);
	cout << "# Initialize b sucessfully!" << endl;
	*/
	b.setOnes();
	
	// in case elem in A is large, we simpy normalize them by divide by the largest elem
	// Normalize(A);
	double end_read_data_time = CycleTimer::currentSeconds();
	// printf("Read data consumes %lf s\n", end_read_data_time - start_read_data_time);	

	this->SetBM_AFromEigen(this->BM_A, this->A, this->d, this->N);
}

IterativeSolver::IterativeSolver(string file_path, int d, int max_step /*= 1000*/, double res_THOLD /*= 1e-20*/, double res_diff_THOLD /*= 0*/){
	ls = LinearSystem(file_path, d, max_step, res_THOLD, res_diff_THOLD);
}

IterativeSolver::IterativeSolver(Eigen::MatrixXd A, Eigen::VectorXd b, int d){
	ls = LinearSystem(A, b, d);
}

Eigen::VectorXd IterativeSolver::mult(Eigen::MatrixXd A, int m, Eigen::VectorXd x){
	int n = A.cols();
	VectorXd ans(n);
	ans.setZero();
	for(int i = 0; i < n; i++){
		for(int idx = lower(i + 1, ls.N, ls.d); idx <= upper(i + 1, ls.N, ls.d); idx++){
			ans.coeffRef(i) += A.coeff(i, idx - 1) * x.coeff(idx - 1);
		}
	}
	return ans;
}

void IterativeSolver::MR(){
	cout << "MRSM" << endl;
	Eigen::setNbThreads(1);

	Eigen::VectorXd r = ls.b - mult(ls.A, ls.m, ls.x);
	Eigen::VectorXd prev_r = ls.b - ls.b;

	cout << 0 << "\t" << r.dot(r) << endl;
	double start_ite_time = CycleTimer::currentSeconds();

	for(int i = 0; i < ls.max_step; i++){
		auto diff = r - prev_r;
		if(r.dot(r) <= ls.res_THOLD || diff.dot(diff) <= ls.res_diff_THOLD)
			break;
		prev_r = r;

		// O(N * m) since A is banded-matrix
		Eigen::VectorXd Ar = mult(ls.A, ls.m, r);
		// O(2 * N)
		double alpha = r.dot(Ar) / Ar.dot(Ar);
		r = r - alpha * Ar;
		cout << i+1 << "\t" << r.dot(r) << endl;
	}

	double end_ite_time = CycleTimer::currentSeconds();
	cout << "# iterates cost: " << end_ite_time - start_ite_time << endl;
}

void IterativeSolver::RowAdjustMR(){
	Eigen::VectorXd r = ls.b - ls.A * ls.x;

	auto A_Diag_vec = ls.A.diagonal();
	DiagonalMatrix<double, Dynamic> A_Diag_Inverse = A_Diag_vec.asDiagonal();

	for(int i = 0; i < ls.max_step; i++){
		Eigen::VectorXd Ar = ls.A * r;
		Eigen::VectorXd alpha_vec = r.array() / (2 * Ar).array();
		Eigen::DiagonalMatrix<double, Dynamic> alpha_diag = alpha_vec.asDiagonal();
		r = r - alpha_diag * ls.A * r;
		cout << i << "\t" << r.dot(r) << endl;
	}
}

void IterativeSolver::ColAdjustDiagMR(){
	Eigen::VectorXd r = ls.b - ls.A * ls.x;

	for(int i = 0; i < ls.max_step; i++){
		Eigen::VectorXd A_diag_vec = ls.A.diagonal();
		Eigen::VectorXd alpha_vec = 1 / A_diag_vec.array();
		Eigen::DiagonalMatrix<double, Dynamic> alpha_diag = alpha_vec.asDiagonal();
		r = r - ls.A * alpha_diag * r;
		cout << i << "\t" << r.dot(r) << endl;
	}
}

void IterativeSolver::CGN(){
	std::cout << "CGN" << endl;
	// A^TAx = A^Tb
	Eigen::MatrixXd CGN_A = ls.A.transpose() * ls.A;
	Eigen::VectorXd CGN_b = ls.A.transpose() * ls.b;
	Eigen::VectorXd CGN_r = CGN_b - CGN_A * ls.x;

	Eigen::VectorXd r 		= ls.b - ls.A * ls.x;
	Eigen::VectorXd prev_r = ls.b - ls.b;
	std::cout << 0 << "\t" << r.dot(r) << endl;

	// start iterate

	Eigen::VectorXd direct_vec = CGN_r;	// d0
	double start_ite_time = CycleTimer::currentSeconds();
	for(int ite = 0; ite < ls.max_step; ite++){
		auto diff = r - prev_r;
		if(r.dot(r) <= ls.res_THOLD || diff.dot(diff) <= ls.res_diff_THOLD)
		// if(r.dot(r) <= ls.res_THOLD)
			break;
		prev_r = r;

		Eigen::VectorXd CGN_A_d = CGN_A * direct_vec;
		double CGN_r_dot_r = CGN_r.dot(CGN_r);
		double alpha = CGN_r_dot_r / direct_vec.dot(CGN_A_d);

		ls.x = ls.x + alpha * direct_vec;
		CGN_r = CGN_r - alpha * CGN_A_d;

		double beta = CGN_r.dot(CGN_r) / CGN_r_dot_r;
		direct_vec = CGN_r + beta * direct_vec;

		r = ls.b - ls.A * ls.x;
		std::cout << ite+1 << "\t" << r.dot(r) << "\t" << endl;
	}
	double end_ite_time = CycleTimer::currentSeconds();
	cout << "# iterates cost: " << end_ite_time - start_ite_time << endl;
}

// YOUSEF P210
void IterativeSolver::BI_CG(){
	std::cout << "BICG" << endl;
	Eigen::VectorXd r 		= ls.b - ls.A * ls.x;
	Eigen::VectorXd prev_r 	= ls.b - ls.b;
	std::cout << 0 << "\t" << r.dot(r) << endl;

	Eigen::VectorXd rs = r;	// r_star
	// rs.setOnes();		// 并行的时候将其设为 1，不然矩阵式对称的就完全等于 CG 方法了
	Eigen::VectorXd p	= r;
	Eigen::VectorXd ps 	= rs;	// p_star
	Eigen::VectorXd prev_rs	= rs;

	// start iterate
	double start_ite_time = CycleTimer::currentSeconds();

	for(int ite = 0; ite < ls.max_step; ite++){
		auto diff = r - prev_r;
		if(r.dot(r) <= ls.res_THOLD || diff.dot(diff) <= ls.res_diff_THOLD)
			break;
		prev_r = r;
		prev_rs = rs;

		Eigen::VectorXd A_p = ls.A * p;
		double alpha = r.dot(rs) / A_p.dot(ps);
		r = r - alpha * A_p;
		rs = rs - alpha * ls.A.transpose() * ps;

		double beta = r.dot(rs) / prev_r.dot(prev_rs); 
		p = r + beta * p;
		ps = rs + beta * ps;

		std::cout << ite+1 << "\t" << r.dot(r) << "\t" << endl;
	}
	double end_ite_time = CycleTimer::currentSeconds();
	cout << "# iterates cost: " << end_ite_time - start_ite_time << endl;
}

Eigen::VectorXd IterativeSolver::Parallel_Spike(int recursive_level){
	// 控制第一层 partition, 以免 求逆的 size 过大
	int partition_num = 2;
	if(recursive_level == 1)
		partition_num = ls.N / 200;
	for(; partition_num <= ls.N / (2 * ls.d); partition_num++){
		ls.x.setOnes();
		double forward_partition_time = 0, inverse_time = 0, calculate_V_W_time = 0, calculate_G_time = 0, split_V_W_G_time = 0, construct_redu_sys_time = 0, solve_redu_sys_time = 0, calculate_all_x_time = 0;
		double total_time = CycleTimer::currentSeconds();
		double log_start_time = 0;
		/*
			step 1. 设置分块数量以及线程数, 将 A 与 F 进行分块
				均匀分块, 将 rest 分配到最后一个分块中
				注意分块大小有限制：size of partition diagonal matrix >= d
				size partition number  <= floor(N / d)
				revised: partition_num <= floor(N / 2d)
		*/
		int per_size	= ls.N / partition_num;
		int last_size	= per_size + ls.N % partition_num;	// last partition size

		// 符号参照 Gallopoulos 2016
		vector<Eigen::MatrixXd> D(partition_num);
		vector<Eigen::MatrixXd> B(partition_num - 1), C(partition_num - 1);
		vector<Eigen::VectorXd> F(partition_num);

		log_start_time = CycleTimer::currentSeconds();
		// #pragma omp parallel for num_threads(1)
		for(int i = 0; i < partition_num; i++){
			// C 的下标 offset by 1, A_i 对应着同时处理 B_i 与 C_{i-1}
			// s means start
			pair<int, int> s_D = {i * per_size, i * per_size};
			pair<int, int> s_B = {i * per_size, i * per_size + per_size};
			pair<int, int> s_C = {i * per_size, i * per_size - ls.d};

			// cout << ls.N << "\t" << ls.d << "\t" << s_D.first << "\t" << s_D.second << "\t" << s_B.first << "\t" << s_B.second << "\t" << s_C.first << "\t" << s_C.second << endl;

			if(i != partition_num - 1){
				D[i] = ls.A.block(s_D.first, s_D.second, per_size, per_size);
				F[i] = ls.b.segment(s_D.first, per_size);
			}
			else{
				D[i] = ls.A.block(s_D.first, s_D.second, last_size, last_size);
				F[i] = ls.b.segment(s_D.first, last_size);
			}
			
			if(i != partition_num - 1)
				B[i] = ls.A.block(s_B.first, s_B.second, per_size, ls.d);
			
			if(i != 0 && i != partition_num - 1)
				C[i-1] = ls.A.block(s_C.first, s_C.second, per_size, ls.d);
			if(i == partition_num - 1)
				C[i-1] = ls.A.block(s_C.first, s_C.second, last_size, ls.d);
		}
		forward_partition_time += (CycleTimer::currentSeconds() - log_start_time);

		// step 2.
		/*
			获取 entry of D 的逆，并计算 V, W, 求解 G
		*/
		log_start_time = CycleTimer::currentSeconds();
		#pragma parallel omp for
		for(int i = 0; i <partition_num; i++){
			D[i] = D[i].inverse();
		}
		inverse_time += (CycleTimer::currentSeconds() - log_start_time);

		vector<Eigen::MatrixXd> D_inverse;
		D_inverse.swap(D);

		log_start_time = CycleTimer::currentSeconds();
		for(int i = 0; i < partition_num - 1; i++){
			B[i] = D_inverse[i] * B[i];
			C[i] = D_inverse[i+1] * C[i];	// attention here, 由于 Aj * wj = Cj, 并且 w 与 c 的下标是 offset by -1 的，所以是 D[i+1]
		}
		calculate_V_W_time += (CycleTimer::currentSeconds() - log_start_time);

		vector<Eigen::MatrixXd> V, W;
		V.swap(B);
		W.swap(C);

		Eigen::VectorXd G = ls.x;

		log_start_time = CycleTimer::currentSeconds();
		for(int i = 0; i < partition_num; i++){
			if(i != partition_num - 1)
				G.segment(i * per_size, per_size) = D_inverse[i] * F[i];
			else
				G.segment(i * per_size, last_size) = D_inverse[i] * F[i];
		}
		calculate_G_time += (CycleTimer::currentSeconds() - log_start_time);

		// step 3
		/*
			构建 reduced matrix, 该步骤中，由 P98.5 可知，per_size >= 2 * d 才能够分解，起到 reduced 的效果
			所以，往回修改 step 1 中的 partition_num <= floor(N/2d), if per_size == 2 * d, 除了最后一个块有中间块，其余中间快 size 都为 0
			思考策略：我们最终是要修改 X, 理想情况是，将 V, W, X, G 都分块，其中，X 要用分块的引用，但是实验发现，segement 无法取引用，所以更新 X 时，要再次定位
				那么，策略为：
					V, W, G 这些我还是分开用块来存放，同时，X 的辅助分块向量也先给初始化了放那儿
					X 的更新，单独用一个向量来存放 reduced system 的解，然后用这个解更新分块的 X，再用 P98 的公式更新分块 X 中的 X'
		*/

		// store V^t, V', V^b
		vector<vector<Eigen::MatrixXd>> V_hat(partition_num - 1, vector<Eigen::MatrixXd>(3));
		vector<vector<Eigen::MatrixXd>> W_hat(partition_num - 1, vector<Eigen::MatrixXd>(3));
		vector<vector<Eigen::VectorXd>> G_hat(partition_num, vector<Eigen::VectorXd>(3));
		vector<vector<Eigen::VectorXd>> x_hat(partition_num, vector<Eigen::VectorXd>(3));

		log_start_time = CycleTimer::currentSeconds();
		for(int i = 0; i < partition_num; i++){
			// mid_cnt, 即中间部分的 size
			int m_cnt = (i != partition_num - 1 ? per_size - 2 * ls.d : last_size - 2 * ls.d);

			if(i != partition_num - 1){
				V_hat[i][0] = V[i].block(0, 0, ls.d, ls.d);
				V_hat[i][1] = V[i].block(ls.d, 0, m_cnt, ls.d);
				V_hat[i][2] = V[i].block(ls.d + m_cnt, 0, ls.d, ls.d);
			}

			if(i != 0){
				W_hat[i-1][0] = W[i-1].block(0, 0, ls.d, ls.d);
				W_hat[i-1][1] = W[i-1].block(ls.d, 0, m_cnt, ls.d);
				W_hat[i-1][2] = W[i-1].block(ls.d + m_cnt, 0, ls.d, ls.d);
			}

			G_hat[i][0] = G.segment(i * per_size, ls.d);
			G_hat[i][1] = G.segment(i * per_size + ls.d, m_cnt);
			G_hat[i][2] = G.segment(i * per_size + ls.d + m_cnt, ls.d);
		}
		split_V_W_G_time += (CycleTimer::currentSeconds() - log_start_time);

		// construct reduced system
		Eigen::MatrixXd redu_A = Eigen::MatrixXd::Identity(partition_num * 2 * ls.d, partition_num * 2 * ls.d);
		Eigen::VectorXd redu_G = Eigen::VectorXd::Zero(partition_num * 2 * ls.d);

		log_start_time = CycleTimer::currentSeconds();
		for(int i = 0; i < partition_num; i++){
			redu_G.segment(i * 2 * ls.d, ls.d) = G_hat[i][0];
			redu_G.segment(i * 2 * ls.d + ls.d, ls.d) = G_hat[i][2];

			if(i != partition_num - 1){
				redu_A.block(i * 2 * ls.d, (i + 1) * 2 * ls.d, ls.d, ls.d) = V_hat[i][0];
				redu_A.block(i * 2 * ls.d + ls.d, (i + 1) * 2 * ls.d, ls.d, ls.d) = V_hat[i][2];
			}

			if(i != 0){
				redu_A.block(i * 2 * ls.d, i * 2 * ls.d - ls.d, ls.d, ls.d) = W_hat[i-1][0];
				redu_A.block(i * 2 * ls.d + ls.d, i * 2 * ls.d - ls.d, ls.d, ls.d) = W_hat[i-1][2];
			}
		}
		construct_redu_sys_time += (CycleTimer::currentSeconds() - log_start_time);

		// 此处考虑递归求解
		log_start_time = CycleTimer::currentSeconds();
		Eigen::VectorXd redu_x;
		if(ls.N <= 400)
			redu_x = redu_A.colPivHouseholderQr().solve(redu_G);
		else{
			// Eigen::VectorXd tmp = redu_A.colPivHouseholderQr().solve(redu_G);
			IterativeSolver ite = IterativeSolver(redu_A, redu_G, 3 * ls.d);
			redu_x = ite.Parallel_Spike(recursive_level + 1);
			// cout << (redu_x - tmp).dot(redu_x - tmp) << endl;
		}

		solve_redu_sys_time += (CycleTimer::currentSeconds() - log_start_time);

		// place redu_x into corrospoding position of x_hat
		log_start_time = CycleTimer::currentSeconds();
		for(int i = 0; i < partition_num; i++){
			x_hat[i][0] = redu_x.segment(i * 2 * ls.d, ls.d);
			x_hat[i][2] = redu_x.segment(i * 2 * ls.d + ls.d, ls.d);
		}

		// caculate x_hat[i][1]
		for(int i = 0; i < partition_num; i++){
			if(i == 0)
				x_hat[i][1] = G_hat[i][1] - V_hat[i][1] * x_hat[i+1][0];	
			else if(i == partition_num - 1)
				x_hat[i][1] = G_hat[i][1] - W_hat[i-1][1] * x_hat[i-1][2];
			else
				x_hat[i][1] = G_hat[i][1] - V_hat[i][1] * x_hat[i+1][0] - W_hat[i-1][1] * x_hat[i-1][2];
		}

		// restore x
		for(int i = 0; i < partition_num; i++){
			// mid_cnt, 即中间部分的 size
			int m_cnt = (i != partition_num - 1 ? per_size - 2 * ls.d : last_size - 2 * ls.d);

			int s_pos = i * per_size;
			ls.x.segment(s_pos, ls.d) = x_hat[i][0];
			ls.x.segment(s_pos + ls.d, m_cnt) = x_hat[i][1];
			ls.x.segment(s_pos + ls.d + m_cnt, ls.d) = x_hat[i][2];
		}
		calculate_all_x_time += (CycleTimer::currentSeconds() - log_start_time);

		Eigen::VectorXd res = ls.b - ls.A * ls.x;

		if(recursive_level == 1){
			cout << "partition_num\t\t" << partition_num << endl;
			cout << "residual\t\t" << res.dot(res) << endl;
			cout << "forward_partition_time\t" << forward_partition_time << endl;
			cout << "inverse_time\t\t" <<  inverse_time << endl;
			cout << "calculate_V_W_time\t" << calculate_V_W_time << endl;
			cout << "calculate_G_time\t" <<  calculate_G_time << endl;
			cout << "split_V_W_G_time\t" <<  split_V_W_G_time << endl;
			cout << "construct_redu_sys_time\t" << construct_redu_sys_time << endl;
			cout << "reduced_sys_size\t" << redu_A.rows() << endl;
			cout << "solve_redu_sys_time\t" <<  solve_redu_sys_time << endl;
			cout << "calculate_all_x_time\t" <<  calculate_all_x_time << endl;
			cout << "total_time:\t\t" << CycleTimer::currentSeconds() - total_time << endl;
			cout << "***********************************" << endl;
		}
		if(recursive_level != 1)
			break;
	}
	
	// 因为系数矩阵的 n, d 无法满足加速的效果，故直接计算
	if(ls.N / (2 * ls.d) <= 1){
		Eigen::MatrixXd densA = ls.A.toDense();
		ls.x = densA.colPivHouseholderQr().solve(ls.b);
	}
	return ls.x;
}

void IterativeSolver::MDMRM(){
	int elapsed_cnt = 0;
	std::cout << "MDMRM" << endl;
	Eigen::setNbThreads(1);

	Eigen::VectorXd r 		= ls.b - ls.A * ls.x;
	Eigen::VectorXd prev_r 	= ls.b - ls.b;

	vector<double> de_amplitudes(ls.m, 0);
	vector<double> possibilities(ls.m, 0);

	std::cout << 0 << "\t" << r.dot(r) << endl;

	// Set M in splitting method as I
	auto A_hat = ls.A;

	// start iterate
	double start_ite_time = CycleTimer::currentSeconds();
	double phase1_time = 0, phase2_time = 0, phase3_time = 0, phase4_time = 0, phase5_time = 0, extra_time = 0;
	double start_time = 0, end_time = 0;

	for(int ite = 0; ite < ls.max_step; ite++){
		auto diff = r - prev_r;
		if(r.dot(r) <= ls.res_THOLD)
			break;
		prev_r = r;

		// precondition, compute environment variable
		// int l = rand() % ls.m + 1; 
        // The trick to choose l (method 1)
        // based on our experienment, this trick has few effect than randomly choosen
		int l = 3;
		if(ite + 1 <= ls.m)
			l = ite + 1;
		else{
			// de_amplitude => possiblities => l
			double total = accumulate(de_amplitudes.begin(), de_amplitudes.end(), 0.0);
			for(int i = 0; i < ls.m; i++){
				possibilities[i] = de_amplitudes[i] / total;
			}
			int rand_num = rand() % 100000;
			double tag = double(rand_num) / 100000;
			l = 0;
			double accu = 0;
			while(tag > accu){
				accu += possibilities[l++]; 
			}
		}
		int t = (ls.N - l + 1 + ls.m - 1) / ls.m;

        // The trick to choose l (method 2)
        // randomly choosen 
		l = rand() % ls.m + 1; 
        	
		// step 1. compute hi, construct Nx(m-1) matrix h = [h1^T, h2^T, ..., hN^T]
		start_time = CycleTimer::currentSeconds();
		// vector<Eigen::VectorXd> h(ls.N, Eigen::VectorXd::Zero(ls.m - 1));
		
		Eigen::MatrixXd h(ls.N, ls.m - 1);
		h.setZero();
		for(int i = 0; i < ls.N; i++){
			int x = i + 1;
			// cout << lower(x) << "\t" << upper(x) << "\t" <<endl;
			for(int y = lower(x, ls.N, ls.d); y <= upper(x, ls.N, ls.d); y++){
				// cout << ((y-l) % ls.m == 0 ? 1 : 0) << "\t" << (y - 1) % ls.m + 1 << endl;
				if((y-l) % ls.m == 0){
					continue;
				}
				else{
					// 伪代码中是下标的 1-1 ，此处应该将 下标 转化为 偏移量
					int s_idx = (y - 1) % ls.m + 1;
					h.coeffRef(x-1, s_idx - 1 + (s_idx > l ? -1 : 0)) = A_hat.coeff(x-1, y-1) * r.coeff(y-1);
				}
			}
		}

		Eigen::MatrixXd hT = h.transpose();
		end_time = CycleTimer::currentSeconds();
		phase1_time += (end_time - start_time);

		start_time = CycleTimer::currentSeconds();
		/* step 2. compute 	u = [u1^T, u2^T, ..., Ut^T] (t x 1)
							z = [z1^T, z2^T, ..., zt^T] (t x 1)
							y = [y1^T, y2^T, ..., yt^T] (t x m-1)
							q = [q1^T, q2^T, ..., qt^T] (t x m-1)
		*/
		Eigen::VectorXd u(t), z(t);
		u.setZero();
		z.setZero();
		// Eigen::MatrixXd y(t, ls.m - 1), q(t, ls.m - 1);
		vector<Eigen::VectorXd> y(t, Eigen::VectorXd::Zero(ls.m - 1)),  q(t, Eigen::VectorXd::Zero(ls.m - 1));

		for(int i = 0; i < t; i++){
			int x = i + 1;
			int off_set_idx = ls.m * (x - 1) + l;	
			for(int j = lower(off_set_idx, ls.N, ls.d); j <= upper(off_set_idx, ls.N, ls.d); j++){
				u[x-1] += r.coeff(j - 1) * A_hat.coeff(j - 1, off_set_idx - 1);
				z[x-1] += A_hat.coeff(j - 1, off_set_idx - 1) * A_hat.coeff(j - 1, off_set_idx - 1);
				if(A_hat.coeff(j - 1, off_set_idx - 1) != 0)
					y[x-1] += A_hat.coeff(j - 1, off_set_idx - 1) * h.row(j - 1);
				if(A_hat.coeff(j - 1, off_set_idx - 1) * r.coeff(off_set_idx - 1) != 0)
					q[x-1] += (A_hat.coeff(j - 1, off_set_idx - 1) * r.coeff(off_set_idx - 1)) * h.row(j - 1);
			}
		}

		end_time = CycleTimer::currentSeconds();
		phase2_time += (end_time - start_time);

		// step 3. construct B(m-1 x m-1), Q(m-1 x t), Y(t x m-1), e1(m-1 x 1), e2(t x 1) in arrowhead linear system
		start_time = CycleTimer::currentSeconds();
		// optimization: B 的计算可以优化, 我们在计算 h 的时候，直接构造 h 矩阵, 那么 B = HH^T
		// Eigen::MatrixXd B = Eigen::MatrixXd::Zero(ls.m-1, ls.m-1), Q(ls.m-1, t), Y(t, ls.m-1);
		Eigen::MatrixXd B = hT * h, Q(ls.m - 1, t), Y(t, ls.m - 1);
		Eigen::VectorXd e1 = Eigen::VectorXd::Zero(ls.m - 1), e2(t);

		for(int i = 0; i < ls.N; i++){
			int x = i + 1;
			e1 += r.coeff(x-1) * hT.col(x-1);
		}

		for(int i = 0; i < t; i++){
			int x = i + 1;
			Q.col(x-1) = q[x-1];
			
			/*
			if(z.coeff(x-1) * r.coeff((x-1) * ls.m + l - 1) != 0){
				Y.row(x-1) 	= (1.0 / (z.coeff(x-1) * r.coeff((x-1) * ls.m + l - 1))) * y[x-1];
				e2.coeffRef(x-1) 	= (1.0 / (z.coeff(x-1) * r.coeff((x-1) * ls.m + l - 1))) * u.coeff(x - 1);
			}
			else{
				Y.row(x-1).setZero();
				e2.coeffRef(x-1) = 0;
			}
			*/
			Y.row(x - 1) = y[x - 1];
			e2.coeffRef(x - 1) = u.coeff(x - 1);
		}
		
		end_time = CycleTimer::currentSeconds();
		phase3_time += (end_time - start_time);

		// step 4. solve Arrow head matrix
		start_time = CycleTimer::currentSeconds();
		Eigen::VectorXd rbM_inverse_vec(t);
		for(int i = 0; i < t; i++){
			int x = i + 1;
			if(z.coeff(x-1) * r.coeff((x-1) * ls.m + l - 1) == 0)
				rbM_inverse_vec.coeffRef(x-1) = 1;
			rbM_inverse_vec.coeffRef(x-1) = 1.0 / (z.coeff(x-1) * r.coeff((x-1) * ls.m + l - 1));
		}

		double extra_start_time = CycleTimer::currentSeconds();

		Eigen::MatrixXd C = B - Q * (rbM_inverse_vec.asDiagonal() * Y);
		Eigen::MatrixXd C_Inverse = C.inverse();

		// cout << "B Q RBM_INVERSE, Y" << endl << B << endl << endl << Q << endl << endl << rbM_inverse_vec << endl << endl << Y << endl << endl;
		// cout << "C and C_INVERSE" << endl << endl << C << endl << endl << C_Inverse << endl;

		double extra_end_time = CycleTimer::currentSeconds();
		extra_time += (extra_end_time - extra_start_time);


		// 由于矩阵中的 coeff 可能过大，所以在 上面这个乘法的计算之中，可能溢出造成 nan 的问题
		// 如果没有溢出的问题，那么直接计算即可，否则，构建完整矩阵，在这个完整矩阵基础之上进行求解
		Eigen::VectorXd s, alpha;

		// 处理 Schur 求解溢出
		if( std::isnan(C.coeff(0, 0)) == false && std::isnan(C_Inverse.coeff(0, 0)) == false){
			/* 
			s = C_Inverse * (e1 - Q * rbM_inverse * e2); 
			alpha = -1 * rbM_inverse * Y * C_Inverse * e1 + (rbM_inverse + rbM_inverse * Y * C_Inverse * Q * rbM_inverse) * e2;
			*/
			s = C_Inverse * (e1 - Q * (rbM_inverse_vec.asDiagonal() * e2)); 
			alpha = -1 * rbM_inverse_vec.asDiagonal() * (Y * (C_Inverse * e1)) + 
					rbM_inverse_vec.asDiagonal() * e2 + 
					(rbM_inverse_vec.asDiagonal() * (Y * (C_Inverse * (Q * (rbM_inverse_vec.asDiagonal() * e2)))));
		}
		else{
			elapsed_cnt++;
			Eigen::MatrixXd reducedM(ls.m - 1 + t, ls.m - 1 + t);
			reducedM.setZero();
			reducedM.block(0, 0, ls.m-1, ls.m-1) = B;
			reducedM.block(0, ls.m-1, ls.m-1, t) = Q;
			reducedM.block(ls.m-1, 0, t, ls.m-1) = Y;
			// reducedM.block(ls.m-1, ls.m-1, t, t) = Eigen::MatrixXd::Identity(t, t);
			Eigen::MatrixXd rbM = Eigen::MatrixXd::Identity(t, t);
			for(int i = 0; i < t; i++){
				int x = i + 1;
				rbM.coeffRef(x-1, x-1) = z.coeff(x-1) * r.coeff((x-1) * ls.m + l - 1);
				if(z.coeff(x-1) * r.coeff((x-1) * ls.m + l - 1) == 0)
					rbM.coeffRef(x-1, x-1) = 1;
			}
			reducedM.block(ls.m-1, ls.m-1, t, t) = rbM;

			Eigen::VectorXd sol(ls.m - 1 + t), bias(ls.m - 1 + t);
			bias.segment(0, ls.m-1) = e1;
			bias.segment(ls.m-1, t) = e2;

			sol = reducedM.partialPivLu().solve(bias);
			s = sol.segment(0, ls.m-1);
			alpha = sol.segment(ls.m-1, t);
		}

		// Restore D
		// Note: Since construct diagonal matrix D and multiplication of matrix-vector is expensive
		// so we do not restore D instead of use vector d directly. 
		// bug reported: [lower(i), upper(i)] for i in [0, t) can not fill all N position in some border case.
		Eigen::VectorXd d(ls.N);
		for(int i = 0; i < t + 1; i++){
			int idx = 0;
			for(int j = i * ls.m; j < (i+1) * ls.m && j < ls.N; j++){
				int x = j + 1;
				if((x - l) % ls.m == 0)
					d.coeffRef(x-1) = alpha.coeff(i);
				else
					d.coeffRef(x-1) = s.coeff(idx++);
			}
		}
		// Eigen::MatrixXd D = d.asDiagonal();
		end_time = CycleTimer::currentSeconds();
		phase4_time += (end_time - start_time);
		
		/*	
		// inspect whether derivative is zero
		// for s
		Eigen::VectorXd derivative_s = Eigen::VectorXd::Zero(ls.m - 1);
		for(int i = 0; i < ls.N; i++){
			int x = i + 1;
			int Li = floor(float(x + ls.d - 1 - l) / ls.m) * ls.m + l;
			int Li_hat = (Li + ls.m - l) / ls.m;
			if(Li > upper(i) || Li < lower(i))
				continue;
			derivative_s += (r.coeff(x - 1) * h[x-1] - 
						h[x-1].transpose() * s * h[x-1] - 
						A_hat.coeff(x-1, Li - 1) * r.coeff(Li - 1) * alpha.coeff(Li_hat - 1) * h[x - 1]);
		}

		Eigen::MatrixXd reducedM(ls.m - 1 + t, ls.m - 1 + t);
		reducedM.setZero();
		reducedM.block(0, 0, ls.m-1, ls.m-1) = B;
		reducedM.block(0, ls.m-1, ls.m-1, t) = Q;
		reducedM.block(ls.m-1, 0, t, ls.m-1) = Y;
		// reducedM.block(ls.m-1, ls.m-1, t, t) = Eigen::MatrixXd::Identity(t, t);
		Eigen::MatrixXd rbM = Eigen::MatrixXd::Identity(t, t);
		for(int i = 0; i < t; i++){
			int x = i + 1;
			rbM.coeffRef(x-1, x-1) = z.coeff(x-1) * r.coeff((x-1) * ls.m + l - 1);
			if(z.coeff(x-1) * r.coeff((x-1) * ls.m + l - 1) == 0)
				rbM.coeffRef(x-1, x-1) = 1;
		}
		reducedM.block(ls.m-1, ls.m-1, t, t) = rbM;
		// cout << B << endl << endl << rbM << endl << endl;

		Eigen::VectorXd sol(ls.m - 1 + t), bias(ls.m - 1 + t);
		sol.setZero();
		bias.setZero();
		sol.segment(0, ls.m-1) = s;
		bias.segment(0, ls.m-1) = e1;
		sol.segment(ls.m-1, t) = alpha;
		bias.segment(ls.m-1, t) = e2;

		Eigen::VectorXd res = bias - reducedM * sol;
		cout << 1 << endl;
		// cout << B << endl << endl << B * B.inverse() << endl;
		
		// 还是在验证，我直接对这个线性系统求解
		sol = reducedM.partialPivLu().solve(bias);
		s = sol.segment(0, ls.m-1);
		alpha = sol.segment(ls.m-1, t);
		res = bias - reducedM * sol;

		// cout << reducedM << endl;
		for(int i = 0; i < reducedM.rows(); i++){
			for(int j = 0; j < reducedM.cols(); j++){
				if(abs(reducedM.coeff(i, j)) < 0.01)
					reducedM.coeffRef(i, j) = 0;
			}
		}

		for(int i = 0; i < t; i++){
			int idx = 0;
			for(int j = i * ls.m; j < (i+1) * ls.m && j < ls.N; j++){
				int x = j + 1;
				if((x - l) % ls.m == 0)
					d.coeffRef(x-1) = alpha.coeff(i);
				else
					d.coeffRef(x-1) = s.coeff(idx++);
			}
		}
		D = d.asDiagonal();
		*/	

		// step 5. statistic
		start_time = CycleTimer::currentSeconds();
		// r = r - A_hat * (D * r);	
		// r = r - A_hat * d.asDiagonal() * r;
		r = r - mult(A_hat, ls.m, d.asDiagonal() * r);
		if( std::isnan(r.coeff(0)) == true ){
			r = prev_r;
			ite--;
			continue;
		}

		// 更新 de_amplitude
		de_amplitudes[l - 1] = prev_r.dot(prev_r) / r.dot(r);

		end_time = CycleTimer::currentSeconds();
		phase5_time += (end_time - start_time);
		std::cout << ite+1 << "\t" << r.dot(r) << "\t" << endl;
	}
	
	cout << "#\t" << elapsed_cnt << endl;
	double end_ite_time = CycleTimer::currentSeconds();
	std::cout << "# iterats cost: " <<  end_ite_time - start_ite_time << endl;
	cout << "# " << phase1_time << "\t" << phase2_time << "\t" << phase3_time << "\t" << phase4_time << "\t" << phase5_time << "\t" << extra_time << endl;
}

void IterativeSolver::JACOBI(){
	std::cout << "JACOBI" << endl;

	Eigen::VectorXd r 		= ls.b - ls.A * ls.x;
	Eigen::VectorXd prev_r 	= ls.b - ls.b;

	std::cout << 0 << "\t" << r.dot(r) << endl;

	// spliting matrix
	Eigen::VectorXd D_vec = ls.A.diagonal();
	Eigen::MatrixXd D = ls.A.diagonal().asDiagonal();
	for(int i = 0; i < D_vec.size(); i++){
		D_vec.coeffRef(i) = 1.0 / D_vec.coeff(i);
	}
	Eigen::MatrixXd D_invs = D_vec.asDiagonal();
	Eigen::MatrixXd L = MatrixXd(ls.A.triangularView<Lower>());
	L = L - D;
	Eigen::MatrixXd U = MatrixXd(ls.A.triangularView<Upper>());
	U = U - D;

	// start iterate
	double start_ite_time = CycleTimer::currentSeconds();
	for(int ite = 0; ite < ls.max_step; ite++){
		auto diff = r - prev_r;
		if(r.dot(r) <= ls.res_THOLD || diff.dot(diff) <= ls.res_diff_THOLD)
			break;
		prev_r = r;

		ls.x = D_invs * ((-L - U) * ls.x + ls.b);
		r = ls.b - ls.A * ls.x;
		std::cout << ite+1 << "\t" << r.dot(r) << "\t" << endl;
	}

	double end_ite_time = CycleTimer::currentSeconds();
	std::cout << "# iterats cost: " <<  end_ite_time - start_ite_time << endl;
}

void IterativeSolver::PARALLEL_MR(){
	/*
	Eigen::VectorXd result_gpu = BM_V_MULT(ls.BM_A, ls.N, ls.d, ls.x);
 
	Eigen::VectorXd result_cpu = ls.A * ls.x;

	cout << result_gpu.dot(result_gpu) << endl;
	cout << result_cpu.dot(result_cpu) << endl;
	*/

	cout << "PARALLEL_MRSM" << endl;
	Eigen::setNbThreads(1);

	Eigen::VectorXd r 		= ls.b - BM_V_MULT(ls.BM_A, ls.N, ls.d, ls.x);
	Eigen::VectorXd prev_r 	= ls.b - ls.b;
	// Eigen::VectorXd prev_r = ls.b - ls.b;

	cout << 0 << "\t" << r.dot(r) << endl;
	double start_ite_time = CycleTimer::currentSeconds();

	double* rp = (double*) malloc (ls.N * sizeof(double));
	memcpy(rp, r.data(), ls.N * sizeof(double));
	double *Ar_p = (double*) malloc (ls.N * sizeof(double));
	for(int i = 0; i < ls.max_step; i++){
		memset(Ar_p, 0, ls.N * sizeof(double));
		if(dot(rp, rp, ls.N) <= ls.res_THOLD)
			break;

		BM_V_MULT(ls.BM_A.data(), ls.BM_A.size(), rp, Ar_p, ls.N, ls.d);

		double alpha = dot(rp, Ar_p, ls.N) / dot(Ar_p, Ar_p, ls.N);
		Scalar(Ar_p, Ar_p, alpha, ls.N);
		SUB(rp, Ar_p, rp, ls.N);
		cout << i+1 << "\t" << dot(rp, rp, ls.N)  << endl;
        if(dot(rp, rp, ls.N) < ls.res_THOLD)
            break;
	}
	free(rp);
	free(Ar_p);
	double end_ite_time = CycleTimer::currentSeconds();
	cout << "# iterates cost: " << end_ite_time - start_ite_time << endl;
}

void IterativeSolver::PARALLEL_MDMRM(){
	/*
	// 测试 move data from array into matrixXd
	int N = 20;
	vector<double> vec(N);	
	iota(vec.begin(), vec.end(), 0);
	Eigen::MatrixXd test = Eigen::Map<Eigen::MatrixXd>(vec.data(), 4, 5);
	cout << test << endl;
	*/
	int elapsed_cnt = 0;
	std::cout << "MDMRM" << endl;
	Eigen::setNbThreads(1);

	Eigen::VectorXd r 		= ls.b - BM_V_MULT(ls.BM_A, ls.N, ls.d, ls.x);
	Eigen::VectorXd prev_r 	= ls.b - ls.b;

	std::cout << 0 << "\t" << dot(r, r) << endl;

	// Set M in splitting method as I
	vector<double> BM_A_hat = ls.BM_A;
	auto A_hat = ls.A;

	// start iterate
	double start_ite_time = CycleTimer::currentSeconds();
	double phase1_time = 0, phase2_time = 0, phase3_time = 0, phase4_time = 0, phase5_time = 0, extra_time = 0;
	double start_time = 0, end_time = 0;

	for(int ite = 0; ite < ls.max_step; ite++){
		auto diff = r - prev_r;
		if(dot(r, r) <= ls.res_THOLD)
			break;
		prev_r = r;

		// precondition, compute environment variable
		// int l = rand() % ls.m + 1; 
		int l = 9;
		int t = (ls.N - l + 1 + ls.m - 1) / ls.m;

		// step 1. compute hi, construct Nx(m-1) matrix h = [h1^T, h2^T, ..., hN^T]
		start_time = CycleTimer::currentSeconds();

		Eigen::MatrixXd h = compute_H(BM_A_hat, ls.N, ls.d, l, r);
		Eigen::MatrixXd hT = h.transpose();

		end_time = CycleTimer::currentSeconds();
		phase1_time += (end_time - start_time);

		/* step 2. compute 	u = [u1^T, u2^T, ..., Ut^T] (t x 1)
							z = [z1^T, z2^T, ..., zt^T] (t x 1)
							y = [y1^T, y2^T, ..., yt^T] (t x m-1)
							q = [q1^T, q2^T, ..., qt^T] (t x m-1)
		*/
		start_time = CycleTimer::currentSeconds();
		Eigen::VectorXd u, z;
		Eigen::MatrixXd y, q;

		// warning: GPU 的精度会低一点？
		copmute_STEP2(BM_A_hat, r, hT, u, z, y, q, ls.N, t, ls.d, l);
		
		end_time = CycleTimer::currentSeconds();
		phase2_time += (end_time - start_time);

		// 此处过大也会出现精度问题
		// step 3. construct B(m-1 x m-1), Q(m-1 x t), Y(t x m-1), e1(m-1 x 1), e2(t x 1) in arrowhead linear system
		start_time = CycleTimer::currentSeconds();
		Eigen::MatrixXd B	= MM_LElem_MComp(hT, h);
		Eigen::MatrixXd Q 	= q.transpose();
		Eigen::MatrixXd Y 	= y;
		Eigen::VectorXd e2 	= u;
		Eigen::VectorXd e1 = compute_e1(h, r, ls.N, ls.m);
		end_time = CycleTimer::currentSeconds();
		phase3_time += (end_time - start_time);

		// step 4. Solve Arrow head matrix
		start_time = CycleTimer::currentSeconds();

		// t x 1
		Eigen::VectorXd rbM_inverse_vec = compute_rbM_inverse_vec(z, r, ls.N, ls.m, t, l);
		Eigen::MatrixXd C = B - MM_LElem_MComp(Q, RowTrans(rbM_inverse_vec, Y));
		Eigen::MatrixXd C_Inverse = C.inverse();


		Eigen::VectorXd s, alpha;
		Eigen::VectorXd row_trans_e2 = RowTrans(rbM_inverse_vec, e2);
		s 		= C_Inverse * (e1 - MM_LElem_MComp(Q, row_trans_e2));
		alpha 	= -1 * RowTrans(rbM_inverse_vec, MM_MElem_LComp(Y, C_Inverse * e1)) + 
				row_trans_e2 + 
				RowTrans(rbM_inverse_vec, MM_MElem_LComp(Y, C_Inverse * MM_LElem_MComp(Q, row_trans_e2)));

		double extra_start_time = CycleTimer::currentSeconds();
		double extra_end_time = CycleTimer::currentSeconds();
		extra_time += (extra_end_time - extra_start_time);

		// Restore D
		Eigen::VectorXd D_vec = RestoreD(s, alpha, ls.N, ls.d, l, ls.m, t);

		end_time = CycleTimer::currentSeconds();
		phase4_time += (end_time - start_time);
		
		// step 5. statistic
		start_time = CycleTimer::currentSeconds();
		r = r - BM_V_MULT(BM_A_hat, ls.N, ls.d, RowTrans(D_vec, r));

		end_time = CycleTimer::currentSeconds();
		phase5_time += (end_time - start_time);
		std::cout << ite+1 << "\t" << r.dot(r) << "\t" << endl;
	}	

	cout << "#\t" << elapsed_cnt << endl;
	double end_ite_time = CycleTimer::currentSeconds();
	std::cout << "# iterats cost: " <<  end_ite_time - start_ite_time << endl;
	cout << "# " << phase1_time << "\t" << phase2_time << "\t" << phase3_time << "\t" << phase4_time << "\t" << phase5_time << "\t" << extra_time << endl;

}

void IterativeSolver::PARALLEL_MDMRM_WITHOUT_EIGEN(){
	int elapsed_cnt = 0;
	std::cout << "PARALLEL_MDMRM" << endl;
	Eigen::setNbThreads(1);

	Eigen::VectorXd r 		= ls.b - BM_V_MULT(ls.BM_A, ls.N, ls.d, ls.x);
	Eigen::VectorXd prev_r 	= ls.b - ls.b;

	std::cout << 0 << "\t" << dot(r, r) << endl;

	// Set M in splitting method as I
	vector<double> BM_A_hat = ls.BM_A;
	auto A_hat = ls.A;

	// start iterate
	double start_ite_time = CycleTimer::currentSeconds();
	double phase1_time = 0, phase2_time = 0, phase3_time = 0, phase4_time = 0, phase5_time = 0, extra_time = 0;
	double start_time = 0, end_time = 0;

	vector<double> de_amplitudes(ls.m, 0);
	vector<double> possibilities(ls.m, 0);

	double *rp;
	double* prev_r_p;

	rp = (double*) malloc (ls.N * sizeof(double));
	prev_r_p = (double*) malloc (ls.N * sizeof(double));

	memcpy(rp, r.data(), ls.N * sizeof(double));
	memset(prev_r_p, 0, ls.N * sizeof(double));


	double *HT_p;
	double *Hp;
	double *up, *zp, *yT_p, *qT_p;
	double *BT_p;
	double *Qp, *Yp, *e1_p, *e2_p, *QT_p, *YT_p;
	double *rbM_inverse_vec_p;
	double *rbM_I_Y_p;
	double *Q_rbM_I_Y_T_p;
	double *rbM_I_e2_p;
	double *Q_rbM_I_e2_p;
	double *Y_C_I_e1_p;
	double *rbM_I_Y_C_I_e1_p;
	double *Y_C_I_Q_rbM_I_e2_p;
	double *rbM_I_Y_C_I_Q_rbM_I_e2_p;
	double *alpha_p;
	double *D_vec_p;
	double *D_r_p;
	double *BM_A_hat_D_r_p;
	Hp		= (double*) malloc (ls.N * (ls.m - 1) * sizeof(double));
	HT_p	= (double*) malloc ((ls.m - 1) * ls.N * sizeof(double));
	BT_p	= (double*) malloc ((ls.m - 1) * (ls.m - 1) * sizeof(double));
	e1_p 	= (double*) malloc ((ls.m - 1) * sizeof(double));
	Q_rbM_I_Y_T_p	= (double*) malloc ((ls.m - 1) * (ls.m - 1) * sizeof(double));
	Q_rbM_I_e2_p	= (double*) malloc ((ls.m - 1) * sizeof(double));
	D_vec_p = (double*) malloc (ls.N * sizeof(double));
	D_r_p	= (double*) malloc (ls.N * sizeof(double));
	BM_A_hat_D_r_p	= (double*) malloc (ls.N * sizeof(double));

	for(int ite = 0; ite < ls.max_step; ite++){
		memset(HT_p,	0, ls.N * (ls.m - 1) * sizeof(double));
		memset(Hp, 		0, ls.N * (ls.m - 1) * sizeof(double));
		memset(BT_p, 	0, (ls.m - 1) * (ls.m - 1) * sizeof(double));
		memset(e1_p,	0, (ls.m - 1) * sizeof(double));
		memset(Q_rbM_I_Y_T_p, 	0, (ls.m - 1) * (ls.m - 1) * sizeof(double));
		memset(Q_rbM_I_e2_p, 	0, (ls.m - 1) * sizeof(double));
		memset(D_vec_p, 0, ls.N * sizeof(double));
		memset(D_r_p, 	0, ls.N * sizeof(double));
		memset(BM_A_hat_D_r_p,	0, ls.N * sizeof(double));

		// auto diff = r - prev_r;
		// if(dot(rp, rp, ls.N) <= ls.res_THOLD)
		// 	break;

		// precondition, compute environment variable
        // The trick to choose l (method 1)
        // based on our experienment, this trick has few effect than randomly choosen
		int l = 1;
		if(ite + 1 <= ls.m)
			l = ite + 1;
		else{
			// de_amplitude => possiblities => l
			double total = accumulate(de_amplitudes.begin(), de_amplitudes.end(), 0.0);
			for(int i = 0; i < ls.m; i++){
				possibilities[i] = de_amplitudes[i] / total;
			}
			int rand_num = rand() % 100000;
			double tag = double(rand_num) / 100000.0;
			l = 0;
			double accu = 0;
			while(tag > accu){
				accu += possibilities[l++]; 
			}
		}
		
        // The trick to choose l (method 2)
        // randomly choosen 
		l = rand() % ls.m + 1; 
        
		int t = (ls.N - l + 1 + ls.m - 1) / ls.m;

		double extra_start_time = CycleTimer::currentSeconds();
		up =	(double*) malloc (t * sizeof(double));
		zp = 	(double*) malloc (t * sizeof(double));
		yT_p = 	(double*) malloc ((ls.m - 1) * t * sizeof(double));
		qT_p = 	(double*) malloc ((ls.m - 1) * t * sizeof(double));
		Yp = 	(double*) malloc (t * (ls.m - 1) * sizeof(double));
		QT_p = 	(double*) malloc (t * (ls.m - 1) * sizeof(double));
		rbM_inverse_vec_p =	(double*) malloc (t * sizeof(double));
		rbM_I_Y_p =	(double*) malloc (t * (ls.m - 1) * sizeof(double));
		rbM_I_e2_p =	(double*) malloc (t * sizeof(double));
		Y_C_I_e1_p =	(double*) malloc (t * sizeof(double));
		rbM_I_Y_C_I_e1_p =	(double*) malloc (t * sizeof(double));
		Y_C_I_Q_rbM_I_e2_p =	(double*) malloc (t * sizeof(double));
		rbM_I_Y_C_I_Q_rbM_I_e2_p =	(double*) malloc (t * sizeof(double));
		alpha_p =	(double*) malloc (t * sizeof(double));
		memset(up,	0, t * sizeof(double));
		memset(zp,	0, t * sizeof(double));
		memset(yT_p,	0, (ls.m - 1) * t * sizeof(double));
		memset(qT_p,	0, (ls.m - 1) * t * sizeof(double));
		// memset(Yp,	0, t * (ls.m - 1) * sizeof(double));
		// memset(QT_p,	0, t * (ls.m - 1) * sizeof(double));
		memset(rbM_inverse_vec_p,	0, t * sizeof(double));
		memset(rbM_I_Y_p,	0, t * (ls.m - 1) * sizeof(double));
		memset(rbM_I_e2_p,	0, t * sizeof(double));
		memset(Y_C_I_e1_p,	0, t * sizeof(double));
		memset(rbM_I_Y_C_I_e1_p,	0, t * sizeof(double));
		memset(Y_C_I_Q_rbM_I_e2_p,	0, t * sizeof(double));
		memset(rbM_I_Y_C_I_Q_rbM_I_e2_p,	0, t * sizeof(double));
		memset(alpha_p,	0, t * sizeof(double));
		double extra_end_time = CycleTimer::currentSeconds();
		extra_time += (extra_end_time - extra_start_time);

		// step 1. compute hi, construct Nx(m-1) matrix h = [h1^T, h2^T, ..., hN^T]
		start_time = CycleTimer::currentSeconds();
		compute_H(HT_p, BM_A_hat.data(), BM_A_hat.size(), ls.N, ls.d, l, rp);
		Trans(HT_p, Hp, ls.m - 1, ls.N);

		end_time = CycleTimer::currentSeconds();
		phase1_time += (end_time - start_time);

		/* step 2. compute 	u = [u1^T, u2^T, ..., Ut^T] (t x 1)
							z = [z1^T, z2^T, ..., zt^T] (t x 1)
							y = [y1^T, y2^T, ..., yt^T] (t x m-1)
							q = [q1^T, q2^T, ..., qt^T] (t x m-1)
		*/
		start_time = CycleTimer::currentSeconds();

		// warning: GPU 的精度会低一点？
		copmute_STEP2(BM_A_hat.data(), BM_A_hat.size(), rp, HT_p, up, zp, yT_p, qT_p, ls.N, t, ls.d, l);
		
		end_time = CycleTimer::currentSeconds();
		phase2_time += (end_time - start_time);

		// step 3. construct B(m-1 x m-1), Q(m-1 x t), Y(t x m-1), e1(m-1 x 1), e2(t x 1) in arrowhead linear system
		// 此处过大也会出现精度问题
		start_time = CycleTimer::currentSeconds();
		MM_LElem_MComp(Hp, Hp, BT_p, ls.m - 1, ls.N, ls.m - 1);
		Eigen::MatrixXd B	= Eigen::Map<Eigen::MatrixXd>(BT_p, ls.m - 1, ls.m-1).transpose();
		Qp = qT_p;
		Trans(yT_p, Yp, ls.m - 1, t);
		YT_p = yT_p;
		e2_p = up;
		compute_e1(Hp, rp, e1_p, ls.N, ls.m);

		Trans(Qp, QT_p, ls.m - 1, t);
		Eigen::VectorXd e1 	= Eigen::Map<Eigen::VectorXd>(e1_p, ls.m - 1);

		end_time = CycleTimer::currentSeconds();
		phase3_time += (end_time - start_time);

		// step 4. Solve Arrow head matrix
		start_time = CycleTimer::currentSeconds();

		// t x 1
		compute_rbM_inverse_vec(zp, rp, rbM_inverse_vec_p, ls.N, ls.m, t, l);

		RowTrans(rbM_inverse_vec_p, Yp, rbM_I_Y_p, t, ls.m - 1);

		MM_LElem_MComp(QT_p, rbM_I_Y_p, Q_rbM_I_Y_T_p, ls.m - 1, t, ls.m - 1);
		// based on experimental result, Q_rbM_I_Y is symmetry
		Eigen::MatrixXd Q_rbM_I_Y_mtx = Eigen::Map<Eigen::MatrixXd>(Q_rbM_I_Y_T_p, ls.m - 1, ls.m - 1).transpose();

		Eigen::MatrixXd C = B - Q_rbM_I_Y_mtx;
		Eigen::MatrixXd C_Inverse = C.inverse();

		Eigen::VectorXd s;
		RowTrans(rbM_inverse_vec_p, e2_p, rbM_I_e2_p, t, 1);

		MM_LElem_MComp(QT_p, rbM_I_e2_p, Q_rbM_I_e2_p, ls.m - 1, t, 1);
		Eigen::VectorXd Q_rbM_I_e2 = Eigen::Map<Eigen::VectorXd>(Q_rbM_I_e2_p, ls.m - 1);

		Eigen::VectorXd C_Inverse_e1_vec = C_Inverse * e1;
		Eigen::VectorXd C_Inverse_Q_rbM_I_e2_vec = C_Inverse * Q_rbM_I_e2;
		s = C_Inverse_e1_vec - C_Inverse_Q_rbM_I_e2_vec;

		MM_MElem_LComp(YT_p, C_Inverse_e1_vec.data(), Y_C_I_e1_p, t, ls.m - 1, 1);

		RowTrans(rbM_inverse_vec_p, Y_C_I_e1_p, rbM_I_Y_C_I_e1_p, t, 1);

		MM_MElem_LComp(YT_p, C_Inverse_Q_rbM_I_e2_vec.data(), Y_C_I_Q_rbM_I_e2_p, t, ls.m - 1, 1);

		RowTrans(rbM_inverse_vec_p, Y_C_I_Q_rbM_I_e2_p, rbM_I_Y_C_I_Q_rbM_I_e2_p, t, 1);

		Scalar(rbM_I_Y_C_I_e1_p, alpha_p, -1, t);
		ADD(alpha_p, rbM_I_e2_p, alpha_p, t);
		ADD(alpha_p, rbM_I_Y_C_I_Q_rbM_I_e2_p, alpha_p, t);


		// Restore D
		RestoreD(s.data(), alpha_p, D_vec_p, ls.N, ls.d, l, ls.m, t);

		end_time = CycleTimer::currentSeconds();
		phase4_time += (end_time - start_time);
		
		// step 5. statistic
		start_time = CycleTimer::currentSeconds();
		RowTrans(D_vec_p, rp, D_r_p, ls.N, 1);
		BM_V_MULT(BM_A_hat.data(), BM_A_hat.size(), D_r_p, BM_A_hat_D_r_p, ls.N, ls.d);
		memcpy(prev_r_p, rp, ls.N * sizeof(double));
		SUB(rp, BM_A_hat_D_r_p, rp, ls.N);

		// 更新 de_amplitude
        double dot_r = dot(rp, rp, ls.N);
		de_amplitudes[l - 1] = dot(prev_r_p, prev_r_p, ls.N) / dot_r;

		end_time = CycleTimer::currentSeconds();
		phase5_time += (end_time - start_time);
		std::cout << ite + 1 << "\t" << dot_r << endl;

		// Free
		free(up);
		free(zp);
		free(yT_p);
		free(qT_p);
		free(Yp);
		free(rbM_inverse_vec_p);
		free(QT_p);
		free(rbM_I_Y_p);
		free(rbM_I_e2_p);
		free(Y_C_I_e1_p);
		free(rbM_I_Y_C_I_e1_p);
		free(Y_C_I_Q_rbM_I_e2_p);
		free(alpha_p);

        if(dot_r <= ls.res_THOLD)
			break;
	}	

	free(rp);
	free(prev_r_p);

	free(Hp);
	free(HT_p);
	free(BT_p);
	free(e1_p);
	free(Q_rbM_I_Y_T_p);
	free(Q_rbM_I_e2_p);
	free(D_vec_p);
	free(D_r_p);
	free(BM_A_hat_D_r_p);

	std::cout << "#\t" << elapsed_cnt << endl;
	double end_ite_time = CycleTimer::currentSeconds();
	std::cout << "# iterats cost: " <<  end_ite_time - start_ite_time << endl;
	std::cout << "# " << phase1_time << "\t" << phase2_time << "\t" << phase3_time << "\t" << phase4_time << "\t" << phase5_time << "\t" << extra_time << endl;

}

void IterativeSolver::PARALLEL_CGN(){
	std::cout << "PARALLEL_CGN" << endl;
	// A^TAx = A^Tb
	vector<double> BM_ATA;
	int ATA_d = 1 + (ls.d - 1) * 2;
	SpMat CGN_A = ls.A.transpose() * ls.A;
	ls.SetBM_AFromEigen(BM_ATA, CGN_A, ATA_d, ls.N);

	Eigen::VectorXd CGN_b = ls.A.transpose() * ls.b;
	Eigen::VectorXd CGN_r = CGN_b - CGN_A * ls.x;

	double* CGN_b_p = (double*) malloc (ls.N * sizeof(double));
	double* CGN_r_p = (double*) malloc (ls.N * sizeof(double));
	double* rp = (double*) malloc (ls.N * sizeof(double));
	double* direct_vec_p = (double*) malloc (ls.N * sizeof(double));
	double* CGN_A_d_p = (double*) malloc (ls.N * sizeof(double));
	double* tmp = (double*) malloc (ls.N * sizeof(double));

	memcpy(CGN_b_p, CGN_b.data(), ls.N * sizeof(double));
	memcpy(CGN_r_p, CGN_r.data(), ls.N * sizeof(double));

	Eigen::VectorXd r 		= ls.b - ls.A * ls.x;
	Eigen::VectorXd prev_r	= ls.b - ls.b;
	memcpy(rp, r.data(), ls.N * sizeof(double));

	std::cout << 0 << "\t" << r.dot(r) << endl;

	// start iterate

	Eigen::VectorXd direct_vec = CGN_r;	// d0
	memcpy(direct_vec_p, direct_vec.data(), ls.N * sizeof(double));

	double start_ite_time = CycleTimer::currentSeconds();
	for(int ite = 0; ite < ls.max_step; ite++){
		if(dot(rp, rp, ls.N) <= ls.res_THOLD)
			break;

		// Eigen::VectorXd CGN_A_d = CGN_A * direct_vec;
		BM_V_MULT(BM_ATA.data(), BM_ATA.size(), direct_vec_p, CGN_A_d_p, ls.N, ATA_d);
		// double CGN_r_dot_r = CGN_r.dot(CGN_r);
		double CGN_r_dot_r = dot(CGN_r_p, CGN_r_p, ls.N);
		// double alpha = CGN_r_dot_r / direct_vec.dot(CGN_A_d);
		double alpha = CGN_r_dot_r	/	dot(direct_vec_p, CGN_A_d_p, ls.N);

		// ls.x = ls.x + alpha * direct_vec;
		Scalar(direct_vec_p, tmp, alpha, ls.N);
		ADD(ls.x.data(), tmp, tmp, ls.N);
		ls.x = Eigen::Map<Eigen::VectorXd>(tmp, ls.N);

		// CGN_r = CGN_r - alpha * CGN_A_d;
		Scalar(CGN_A_d_p, tmp, alpha, ls.N);
		SUB(CGN_r_p, tmp, CGN_r_p, ls.N);

		// double beta = CGN_r.dot(CGN_r) / CGN_r_dot_r;
		double beta = dot(CGN_r_p, CGN_r_p, ls.N) / CGN_r_dot_r;
		// direct_vec = CGN_r + beta * direct_vec;
		Scalar(direct_vec_p, tmp, beta, ls.N);
		ADD(CGN_r_p, tmp, direct_vec_p, ls.N);

		// r = ls.b - ls.A * ls.x;
		BM_V_MULT(ls.BM_A.data(), ls.BM_A.size(), ls.x.data(), tmp, ls.N, ls.d);
		SUB(ls.b.data(), tmp, rp, ls.N);

		std::cout << ite+1 << "\t" << dot(rp, rp, ls.N) << "\t" << endl;
	}

	free(rp);
	free(CGN_b_p);
	free(CGN_r_p);
	free(direct_vec_p);
	free(CGN_A_d_p);
	free(tmp);

	double end_ite_time = CycleTimer::currentSeconds();
	cout << "# iterates cost: " << end_ite_time - start_ite_time << endl;
}

void IterativeSolver::PARALLEL_BICG(){
	std::cout << "PARALLEL_BICG" << endl;
	Eigen::VectorXd r 		= ls.b - ls.A * ls.x;
	Eigen::VectorXd prev_r 	= ls.b - ls.b;

	Eigen::VectorXd rs 	= r;	// r_star
	Eigen::VectorXd p	= r;
	Eigen::VectorXd ps 	= rs;	// p_star
	Eigen::VectorXd prev_rs	= rs;
	std::cout << 0 << "\t" << r.dot(r) << endl;

	double *rp 			= (double*) malloc (ls.N * sizeof(double));
	double *prev_r_p 	= (double*) malloc (ls.N * sizeof(double));
	double *rs_p 		= (double*) malloc (ls.N * sizeof(double));
	double *p_p 		= (double*) malloc (ls.N * sizeof(double));
	double *ps_p 		= (double*) malloc (ls.N * sizeof(double));
	double *prev_rs_p	= (double*) malloc (ls.N * sizeof(double));
	double *tmp	= (double*) malloc (ls.N * sizeof(double));
	vector<double> BM_A_T;

	memcpy(rp, r.data(), ls.N * sizeof(double));
	memcpy(prev_r_p, prev_r.data(), ls.N * sizeof(double));
	memcpy(rs_p, rs.data(), ls.N * sizeof(double));
	memcpy(p_p, p.data(), ls.N * sizeof(double));
	memcpy(ps_p, ps.data(), ls.N * sizeof(double));
	memcpy(prev_rs_p, prev_rs.data(), ls.N * sizeof(double));

	SpMat AT = ls.A.transpose();
	ls.SetBM_AFromEigen(BM_A_T, AT, ls.d, ls.N);

	// start iterate
	double start_ite_time = CycleTimer::currentSeconds();
	for(int ite = 0; ite < ls.max_step; ite++){
		if(dot(rp, rp, ls.N) <= ls.res_THOLD)
			break;
		memcpy(prev_r_p, rp, ls.N * sizeof(double));
		memcpy(prev_rs_p, rs_p, ls.N * sizeof(double));

		// Eigen::VectorXd A_p = ls.A * p;
		BM_V_MULT(ls.BM_A.data(), ls.BM_A.size(), p_p, tmp, ls.N, ls.d);
		double alpha = dot(rp, rs_p, ls.N) / dot(tmp, ps_p, ls.N);

		// r = r - alpha * A_p;
		Scalar(tmp, tmp, alpha, ls.N);
		SUB(rp, tmp, rp, ls.N);
		// rs = rs - alpha * ls.A.transpose() * ps;
		BM_V_MULT(BM_A_T.data(), BM_A_T.size(), ps_p, tmp, ls.N, ls.d);
		Scalar(tmp, tmp, alpha, ls.N);
		SUB(rs_p, tmp, rs_p, ls.N);

		double beta = dot(rp, rs_p, ls.N) / dot(prev_r_p, prev_rs_p, ls.N); 
		// p = r + beta * p;
		Scalar(p_p, tmp, beta, ls.N);
		ADD(rp, tmp, p_p, ls.N);
		// ps = rs + beta * ps;
		Scalar(ps_p, tmp, beta, ls.N);
		ADD(rs_p, tmp, ps_p, ls.N);

		std::cout << ite+1 << "\t" << dot(rp, rp, ls.N) << "\t" << endl;
	}

	free(rp);
	free(prev_r_p);
	free(rs_p);
	free(p_p);
	free(ps_p);
	free(prev_rs_p);
	free(tmp);

	double end_ite_time = CycleTimer::currentSeconds();
	cout << "# iterates cost: " << end_ite_time - start_ite_time << endl;
}

void IterativeSolver::PARALLEL_CG(){
	cout << "PARALLEL_CG" << endl;
	Eigen::setNbThreads(1);

	Eigen::VectorXd r = ls.b - ls.A * ls.x;

	cout << 0 << "\t" << r.dot(r) << endl;

	double *rp = 		(double*) malloc (ls.N * sizeof(double));
	double *prev_r_p = 	(double*) malloc (ls.N * sizeof(double));
	double *dp = 		(double*) malloc (ls.N * sizeof(double));
	double *tmp = 		(double*) malloc (ls.N * sizeof(double));

	memcpy(rp, r.data(), ls.N * sizeof(double));
	memcpy(dp, rp, ls.N * sizeof(double));

	double start_ite_time = CycleTimer::currentSeconds();
	for(int i = 0; i < ls.max_step; i++){
		memcpy(prev_r_p, rp, ls.N * sizeof(double));
		BM_V_MULT(ls.BM_A.data(), ls.BM_A.size(), dp, tmp, ls.N, ls.d);
		double alpha = dot(rp, rp, ls.N) / dot(dp, tmp, ls.N);

		Scalar(tmp, tmp, alpha, ls.N);
		SUB(rp, tmp, rp, ls.N);

		double beta = dot(rp, rp, ls.N) / dot(prev_r_p, prev_r_p, ls.N);
		Scalar(dp, tmp, beta, ls.N);
		ADD(rp, tmp, dp, ls.N);

		cout << i+1 << "\t" << dot(rp, rp, ls.N) << endl;
        if(dot(rp, rp, ls.N) <= ls.res_THOLD)
            break;
	}

	free(rp);
	free(prev_r_p);
	free(dp);
	free(tmp);

	double end_ite_time = CycleTimer::currentSeconds();
	cout << "# iterates cost: " << end_ite_time - start_ite_time << endl;
}