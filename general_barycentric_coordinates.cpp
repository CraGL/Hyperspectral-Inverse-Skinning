#include <igl/cotmatrix.h>
#include <igl/read_triangle_mesh.h>
#include <igl/list_to_matrix.h>
#include <igl/readDMAT.h>
#include <igl/REDRUM.h>
#include <igl/active_set.h>
// #include <igl/mosek/mosek_quadprog.h>

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
	bool use_sparse_solver = false)
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
	Eigen::SparseMatrix<double> A(n*h, n*h);
	A.reserve(L.nonZeros()*h);
	for(int i=0; i<h; i++) {
		for (int k=0; k<L.outerSize(); ++k) {
			for (SparseMatrix<double>::InnerIterator it(L,k); it; ++it)
			{
				A.insert(it.row()+n*i,it.col()+n*i) = it.value();
			}
		}
	}

	VectorXd B(n*h);
	B.setZero();
	cout << "Sparse Laplacian system built." << endl;
	cout << "Sparse L: " << L.nonZeros() << endl;
	cout << "Sparse A: " << A.nonZeros() << " " << A.rows() << " " << A.cols() << endl;

	// add soft constraints TE.X + te0 = 0, representing sum_i(w_ij*T_ij) = T_j
	std::vector<Tr> tripletList;
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
	A += A2;
	B += temp*te0*penalty_weight;
	cout << "Soft equality constraints added." << endl;	
	
	// equality constraints, representing sum_i(w_ij) = 1
	tripletList.clear();
	tripletList.reserve(n*h);
	
	for(int i=0; i<h; i++) {
		for(int j=0; j<n; j++) {
			tripletList.push_back(Tr(j,i*n+j,1));
		}
	}
	Eigen::SparseMatrix<double> Aeq(n,n*h);
	Aeq.setFromTriplets(tripletList.begin(), tripletList.end());
	VectorXd Beq(n);
	Beq.setConstant(1);	
	cout << "Equality constraints built." << endl;
	
	// solve
	VectorXd X;
	
	if( use_sparse_solver ) {
	
		VectorXd lx(n*h), ux(n*h);
		lx.setConstant(0);
		ux.setConstant(1);
		cout << "Inequality constraints built." << endl;
	
		VectorXi known;
		VectorXd Y;
		Eigen::SparseMatrix<double> Aieq;
		VectorXd Bieq;
		
		igl::active_set_params params;
		params.Auu_pd = true;
		params.max_iter = 1;
	
		igl::active_set(A,B,known,Y,Aeq,Beq,Aieq,Bieq,lx,ux,params,X);
	}
	else {
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
		solve_quadprog(G,g0,CE,ce0,CI,ci0,X);
	}
	cout << GREENGIN("Weight solved.") << endl;
	
	// reshape new weights
	nW.resize(n,h);
	for(int i=0; i<h; i++) {
		nW.col(i) = X.segment(i*n,n);
	}
	cout << nW << endl;
}

int main(int argc, char* argv[]) {
	using namespace std;
	using namespace Eigen;
	using namespace pythonlike;
	
	vector<string> args( argv + 1, argv + argc );
	string weight_path;
	MatrixXd W;
	const bool found_weight_param = get_optional_parameter( args, "--weight", weight_path );
	if(!igl::readDMAT(weight_path, W)) {
		cerr << "Cannot read weights from " + weight_path << endl;
		exit(-1);
	}
		
	assert(args.size() >= 4 && args.size()%2 == 0);
	vector<Pose> poses;
	
	for(int i=0; i<args.size(); i+=2) {
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
	
	MatrixXd nW;
	bool use_sparse_solver = true;
	Measure<>::execution(solve_weights, poses, W, nW, use_sparse_solver);
	double max_error = (nW-W).array().abs().maxCoeff();
	cout << "max error: " << max_error << endl;
	if( max_error > 1e-4) {
		cout << "Weight Diff:\n" << nW-W << endl;
	}
	
	return 0;
}