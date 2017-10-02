#include <igl/cotmatrix.h>
#include <igl/read_triangle_mesh.h>
#include <igl/list_to_matrix.h>
#include <igl/readDMAT.h>
#include <igl/writeDMAT.h>
#include <igl/REDRUM.h>
#include <igl/active_set.h>
#include <igl/mosek/mosek_quadprog.h>

#include <string>
#include <vector>
#include <iostream>
#include <chrono>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include "pythonlike.h"
#include "eiquadprog.h"

namespace {
	struct Pose{
		Eigen::MatrixXd V;	// #Vx3
		Eigen::MatrixXi F;	// #Fx3
		Eigen::MatrixXd T;	// #Hx12
	};

	enum SolverType{
		mosek=0,
		active_set,
		eiquadprog
	};

	template<typename TimeT = std::chrono::milliseconds>
	struct Measure
	{
		template<typename F, typename ...Args>
		static typename TimeT::rep execution(F&& func, Args&&... args)
		{
			auto start = std::chrono::steady_clock::now();
			std::forward<decltype(func)>(func)(std::forward<Args>(args)...);
			auto duration = std::chrono::duration_cast< TimeT> 
								(std::chrono::steady_clock::now() - start);
			std::cout << "duration: " + std::to_string(duration.count()) + " in milliseconds" << std::endl;
			return duration.count();
		}
	};
}

void read_transform_from_file(const std::string &path, Eigen::MatrixXd & T)
{
	using namespace std;
	
	std::ifstream file;
	file.open(path);
	string line;
	vector< vector<double> > transforms;
	while(getline(file,line))
	{
		vector<double> row(3);
		int count = sscanf(line.c_str(),"%lg %lg %lg",
					  &row[0],
					  &row[1],
					  &row[2]);
		transforms.push_back(row);
		if(count != 3)
		{
			cout << "Error: bad format in vertex line" << endl;
			file.close();
			exit(-1);
		}
	}
	file.close();
	igl::list_to_matrix(transforms,T);
}

void solve_weights(
	const std::vector<Pose> & poses, 
	const Eigen::MatrixXd & W,
	Eigen::MatrixXd & nW,
	SolverType solver_type)
{
	using namespace std;
	using namespace Eigen;
	
	typedef Eigen::Triplet<double> Tr;

	const int p = poses.size();				// #poses
	assert(p>=1);
	const int n = poses[0].V.rows();		// #vertices
	const int h = poses[0].T.rows();		// #handles
	assert( W.cols() == h );
	
	cout << "sparse solver: " << endl;
	cout << "Prepare to solve weights" << endl;
	cout << "System size: " << n*h << endl;
	// Laplacian
	Eigen::SparseMatrix<double> L;
	igl::cotmatrix(poses[0].V,poses[0].F,L);
	
	L = -L;

	std::vector<Tr> tripletList;
	tripletList.reserve(L.nonZeros()*h);
	for(int i=0; i<h; i++) {
		for (int k=0; k<L.outerSize(); ++k) {
			for (SparseMatrix<double>::InnerIterator it(L,k); it; ++it)
			{
				tripletList.push_back(Tr(it.row()+n*i,it.col()+n*i,it.value()));
			}
		}
	}
	Eigen::SparseMatrix<double> A(n*h, n*h), A1(n*h, n*h);
	A1.setFromTriplets(tripletList.begin(), tripletList.end());

	VectorXd B(n*h);
	B.setZero();
	cout << "Sparse Laplacian system built." << endl;
	cout << "Sparse L: " << L.nonZeros() << endl;
	cout << "Sparse A: " << A.nonZeros() << " " << A.rows() << " " << A.cols() << endl;

	// add soft constraints TE.X + te0 = 0, representing sum_i(w_ij*T_ij) = T_j
	tripletList.clear();
	tripletList.reserve(12*h*n);
	
	VectorXd te0(n*12*p);
	for(int i=0; i<p; i++) {
		MatrixXd WT = W*poses[i].T;		// nx12
		for(int k=0; k<n; k++){
			for(int j=0; j<h; j++) {
				for(int t=0; t<12; t++) {
					tripletList.push_back(Tr(i*n*12+k*12+t, j*n+k, poses[i].T(j,t)));
				}
			}
			te0.segment(i*n*12+k*12,12) = -WT.row(k).transpose();
		}
	}
	Eigen::SparseMatrix<double> TE(n*12*p,n*h);
	TE.setFromTriplets(tripletList.begin(), tripletList.end());
// 	cout << "Sparse TE: " << TE.nonZeros() << " " << TE.rows() << " " << TE.cols() << endl;
	
	const double penalty_weight = 1e6;
	Eigen::SparseMatrix<double> temp(TE.transpose());
	Eigen::SparseMatrix<double> A2 = temp*TE;
	A2 *= penalty_weight;

	cout << "Sparse A2: " << A2.nonZeros() << " " << A2.rows() << " " << A2.cols() << endl;
	A = A1 + A2;
	B = temp*te0*penalty_weight;
	cout << "Soft equality constraints added." << endl;	
	
	// equality constraints, representing sum_i(w_ij) = 1
	std::vector<Tr> Aeq_TripList;
	Aeq_TripList.clear();
	Aeq_TripList.reserve(n*h);
	
	for(int i=0; i<h; i++) {
		for(int j=0; j<n; j++) {
			Aeq_TripList.push_back(Tr(j,i*n+j,1));
			tripletList.push_back(Tr(n*12*p+j,i*n+j,1));
		}
	}
	Eigen::SparseMatrix<double> Aeq(n,n*h);
	Aeq.setFromTriplets(Aeq_TripList.begin(), Aeq_TripList.end());
	VectorXd Beq(n);
	Beq.setConstant(1);	
	cout << "Equality constraints built." << endl;
	
	// solve
	VectorXd X;
	
// 	string output_system = "../models/system.DMAT";
// 	igl::writeDMAT(output_system, MatrixXd(A));
	
	if( solver_type == mosek ) {
		VectorXd lx(n*h);
		VectorXd ux(n*h);
		lx.setZero();
		ux.setOnes();
		VectorXd lc, uc;
		SparseMatrix<double> Aieq;
		double cf = 0.;
		
		// convert equality constraints to energy
		Eigen::SparseMatrix<double> QE(n*12*p+n,n*h);
		QE.setFromTriplets(tripletList.begin(), tripletList.end());
		
		Eigen::SparseMatrix<double> temp2(QE.transpose());
		Eigen::SparseMatrix<double> A3 = temp2*QE;
		A3 *= penalty_weight;
		
		VectorXd qe0(n*12*p+n);
		qe0.segment(0,n*12*p) = te0;
		qe0.segment(n*12*p,n) = -Beq;
		
		A = A1 + A3;
		B = temp2*qe0*penalty_weight;
		
		igl::mosek::MosekData mosek_data;
		igl::mosek::mosek_quadprog(A,B,cf,Aieq,lc,uc,lx,ux,mosek_data,X);
		
	} 
	else if( solver_type == active_set ) {
	
// 		VectorXd lx(n*h), ux(n*h);
// 		lx.setConstant(0);
// 		ux.setConstant(1);
// 		Eigen::SparseMatrix<double> Aieq;
// 		VectorXd Bieq;
		
		VectorXd lx, ux;
		tripletList.clear();
		tripletList.reserve(n*h);
	
		for(int i=0; i<n*h; i++) {
			tripletList.push_back(Tr(i,i,1));
		}
		Eigen::SparseMatrix<double> Aieq(n*h,n*h);
		Aieq.setFromTriplets(tripletList.begin(), tripletList.end());
		VectorXd Bieq(n*h);
		Bieq.setZero();
		cout << "Inequality constraints built." << endl;
	
		VectorXi known;
		VectorXd Y;
		
		igl::active_set_params params;
		params.Auu_pd = false;
		params.max_iter = 1;
	
		igl::active_set(A,B,known,Y,Aeq,Beq,Aieq,Bieq,lx,ux,params,X);
	}
	else if( solver_type == eiquadprog ) {
		// w_ij >= 0
		tripletList.clear();
		tripletList.reserve(n*h);
	
		for(int i=0; i<n*h; i++) {
			tripletList.push_back(Tr(i,i,1));
		}
		Eigen::SparseMatrix<double> Aieq(n*h,n*h);
		Aieq.setFromTriplets(tripletList.begin(), tripletList.end());
		VectorXd Bieq(n*h);
		Bieq.setZero();
		cout << "Inequality constraints built." << endl;
		
		MatrixXd G = MatrixXd(A);
		VectorXd g0 = B;
		MatrixXd CE = MatrixXd(Aeq).transpose();
		VectorXd ce0 = -Beq;
		MatrixXd CI = MatrixXd(Aieq).transpose();
		VectorXd ci0 = -Bieq;
		
		Eigen::LLT<Eigen::MatrixXd> lltOfG(G); // compute the Cholesky decomposition of A
		if(lltOfG.info() == Eigen::NumericalIssue)
		{
			throw std::runtime_error("Possibly non positive definite matrix!");
		} 
		
		solve_quadprog(G,g0,CE,ce0,CI,ci0,X);
	} else {
		cerr << "Unknown solver" << endl;
		exit(-1);
	}
	cout << GREENGIN("Weight solved.") << endl;
	
	// reshape new weights
	nW.resize(n,h);
	for(int i=0; i<h; i++) {
		nW.col(i) = X.segment(i*n,n);
	}
}

void solve_weights_locally(
	const std::vector<Pose> & poses, 
	const Eigen::MatrixXd & W,
	Eigen::MatrixXd & nW)
{
	using namespace std;
	using namespace Eigen;
	
	typedef Eigen::Triplet<double> Tr;
	
	const int p = poses.size();				// #poses
	assert(p>=1);
	const int n = poses[0].V.rows();		// #vertices
	const int h = poses[0].T.rows();		// #handles
	assert( W.cols() == h );
	
	// Laplacian
	Eigen::SparseMatrix<double> L;
	igl::cotmatrix(poses[0].V,poses[0].F,L);
	
	L = -L;
// 	Eigen::SparseMatrix<double> A2(n*h, n*h);

	std::vector<Tr> tripletList;
	tripletList.reserve(L.nonZeros()*h);
	
	for(int i=0; i<h; i++) {
		for (int k=0; k<L.outerSize(); ++k) {
			for (SparseMatrix<double>::InnerIterator it(L,k); it; ++it)
			{
				int row = it.row();
				int col = it.col();
				tripletList.push_back(Tr(row+n*i,col+n*i,it.value()/L.coeffRef(row,row)));
// 				tripletList.push_back(Tr(row+n*i,col+n*i,it.value()));
			}
		}
	}
	Eigen::SparseMatrix<double> LL(n*h,n*h);
	LL.setFromTriplets(tripletList.begin(), tripletList.end());

// 	cout << L << endl;
	cout << "#vertices: " << n << endl;
	cout << "#handles: " << h << endl;
	cout << "#poses: " << p << endl;
	
	cout << "Sparse Laplacian system built." << endl;
	cout << "Sparse L: " << L.nonZeros() << " " << L.rows() << " " << L.cols() << endl;
	cout << "Sparse LL: " << LL.nonZeros() << " " << LL.rows() << " " << LL.cols() << endl;
	
	const int max_iter = 1000;
	MatrixXd curr_W(n,h),prev_W(n,h);
	VectorXd smooth_W(n*h);
	curr_W.setConstant(1./h);

// 	curr_W.setZero();
// 	for(int i=0; i<n; i++) 	curr_W(i,i%h) = 1;
	
// 	curr_W.setRandom();
// 	curr_W = (curr_W.array()+1.)/2;
// 	for(int i=0; i<n; i++)	curr_W.row(i) /= curr_W.row(i).sum();
	
	const double scale = 0.25;
	const double threshold = 1e-7;
	
	cout << "curr_W:" << endl;
	cout << curr_W << endl;
	vector<double> weight_diff_vec;
	
	for(int k=0; k<max_iter; k++) {
		
		prev_W = curr_W;
		for(int i=0; i<h; i++)	smooth_W.segment(i*n,n) = curr_W.col(i);
		VectorXd step = LL*smooth_W;
		cout << "iter #" << k << " step: " << step.norm() << endl;
		
		cout << step.rows()  << " " << smooth_W.rows() << endl;
		smooth_W -= scale*step;
		
		for(int i=0; i<n; i++) {
			// minimize w - curr_W
			MatrixXd I(h,h);
			I.setIdentity();
			
			VectorXd i0(h);
			for(int j=0; j<h; j++)	i0(j) = -smooth_W(j*n+i);
// 			i0 = -smooth_W.segment(i*h,h).transpose();
			
			// add soft constraints TE.X + te0 = 0, representing sum_i(w_ij*T_ij) = T_j
			MatrixXd TE(12*p,h);
	
			VectorXd te0(12*p);
			for(int j=0; j<p; j++) {
				TE.block(j*12,0,12,h) = poses[j].T.transpose();
				te0.segment(j*12,12) = -(W.row(i)*poses[j].T).transpose();
			}
	
			const double penalty_weight = 1e6;
			MatrixXd A = TE.transpose()*TE*penalty_weight;

			VectorXd B(h);
			B = TE.transpose()*te0*penalty_weight;
			// equality constraints, representing sum_i(w_ij) = 1
			MatrixXd Aeq(1,h);
			Aeq.setOnes();
			VectorXd Beq(1);
			Beq << 1;
			
			// inequality constraints, w_ij >= 0
			MatrixXd Aieq(h,h);
			Aieq.setIdentity();
			VectorXd Bieq(h);
			Bieq.setZero();
			
			// solve one vertex
			MatrixXd G = I+A;
			VectorXd g0 = i0+B;
			MatrixXd CE = Aeq.transpose();
			VectorXd ce0 = -Beq;
			MatrixXd CI = Aieq.transpose();
			VectorXd ci0 = -Bieq;
			VectorXd X;
			solve_quadprog(G,g0,CE,ce0,CI,ci0,X);
			
			curr_W.row(i) = X.transpose();
// 			if(X.array().maxCoeff() > 1 || X.array().minCoeff() < 0) cout << X.transpose() << endl;
		}
		
// 		cout << curr_W << endl;
		
		double norm = (prev_W-curr_W).norm();
		cout << "weight difference norm: " << norm << endl;
		if( !weight_diff_vec.empty() && abs(norm - weight_diff_vec.back()) <= threshold) {
// 		if(norm <= threshold) {
			cout << "Stop optimization at " << k << " iterations" << endl;
			break;
		}
		weight_diff_vec.push_back(norm);
	}
	nW = curr_W;
}

namespace
{
	void usage( const char* argv0 ) {
		std::cerr<<"Usage:"<<std::endl<<"    " << argv0 << " pos1.obj pos2.obj pos3.obj ... pos1.Tmat pos2.Tmat pos3.Tmat ... \
			[--weights path/to/weigvhts_ground_truth.DMAT]"<< std::endl;
		// exit(0) means success. Anything else means failure.
		exit(-1);
	}
}

int main(int argc, char* argv[]) {
	using namespace std;
	using namespace Eigen;
	using namespace pythonlike;
	
	vector<string> args( argv + 1, argv + argc );
	string weight_path;
	MatrixXd W;
	const bool found_weight_param = get_optional_parameter( args, "--weight", weight_path );
	if(found_weight_param && !igl::readDMAT(weight_path, W)) {
		cerr << "Cannot read weights from " + weight_path << endl;
		usage(argv[0]);
	}
		
	assert(args.size() >= 4 && args.size()%2 == 0);
	vector<Pose> poses;
	
	const int num_poses = args.size()/2;
	for(int i=0; i<num_poses; i++) {
		Pose P;
		igl::read_triangle_mesh(args[i], P.V, P.F);
		MatrixXd T;	
		read_transform_from_file(args[i+1], T);
		assert(T.rows()%4==0 && T.cols()==3);
		int h = T.rows()/4;
		P.T.resize(h, 12);
		for(int i=0; i<h; i++)
			for(int j=0; j<4; j++) {
				P.T.block(i,j*3,1,3) = T.row(i*4+j);
			}
		if(i>0)	{
			assert(P.V.rows() == poses.back().V.rows() && "Poses have the same number of vertices.");
			assert(P.F.rows() == poses.back().F.rows() && "Poses have the same number of faces.");
			assert(P.T.rows() == poses.back().T.rows() && "Poses have the same number of handles.");
		}
		poses.push_back(P);
	}
	
	SolverType solver_type = mosek;
	
	MatrixXd nW;
// 	Measure<>::execution(solve_weights, poses, W, nW, solver_type);
	Measure<>::execution(solve_weights_locally, poses, W, nW);
	cout << nW << endl;
	double max_error = (nW-W).array().abs().maxCoeff();
	cout << "max error: " << max_error << endl;
	
// 	double max_weight = (nW).array().abs().maxCoeff();
// 	cout << "max weight: " << max_weight << endl;
	if( max_error > 1e-4) {
// 		cout << "Weight:\n" << nW << endl;
	}
	
	// normalize weights
	for(int i=0; i<nW.rows(); i++)	nW.row(i) /= nW.row(i).sum();
	
	string output_weight = pythonlike::os_path_split(weight_path).first+"-estimated.DMAT";
	igl::writeDMAT(output_weight, nW);
	
	return 0;
}