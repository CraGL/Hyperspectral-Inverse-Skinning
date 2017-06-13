#include <igl/cotmatrix.h>
#include <igl/read_triangle_mesh.h>
#include <igl/list_to_matrix.h>
#include <igl/readDMAT.h>
#include <igl/REDRUM.h>
#include <igl/active_set.h>

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

void solve_weights_dense(
	const std::vector<Pose> & poses, 
	const Eigen::MatrixXd & W,
	Eigen::MatrixXd & nW)
{
	using namespace std;
	using namespace Eigen;
	
	const int p = poses.size();				// #poses
	assert(p>=1);
	const int n = poses[0].V.rows();		// #vertices
	const int h = poses[0].T.rows();		// #handles
	assert( W.cols() == h );
	
	cout << "Prepare to solve weights" << endl;
	cout << "System size: " + to_string(n*h) << endl;
	// Laplacian
	Eigen::SparseMatrix<double> sparse_L;
	igl::cotmatrix(poses[0].V,poses[0].F,sparse_L);
	
	MatrixXd L = -MatrixXd(sparse_L);
	assert( L.rows() == n && L.cols() == n );
	MatrixXd G(n*h, n*h);
	for(int i=0; i<h; i++) {
		G.block(i*n, i*n, n, n) = L;
	}
	VectorXd g0(n*h);
	g0.setZero();
	cout << "Sparse Laplacian system built." << endl;
	
	// add soft constraints TE^T.X + te0 = 0, representing sum_i(w_ij*T_ij) = T_j
	MatrixXd TE(n*h,n*12*p);
	VectorXd te0(n*12*p);
	for(int i=0; i<p; i++) {
		MatrixXd TE_i(n*12,n*h);
		TE_i.setZero();
		MatrixXd WT = W*poses[i].T;		// nx12
		for(int k=0; k<n; k++){
			for(int j=0; j<h; j++) {
				TE_i.block(k*12,j*n+k,12,1) = poses[i].T.row(j).transpose();
			}
			te0.segment(i*12*n+k*12,12) = -WT.row(k).transpose();
		}
		TE.block(0,i*12*n,n*h,n*12) = TE_i.transpose();
	}
	const double penalty_weight = 1e6;
	G += TE*TE.transpose()*penalty_weight;
	g0 += TE*te0*penalty_weight;
	cout << "Soft equality constraints added." << endl;
	
	// w_ij >= 0
	MatrixXd CI(n*h,n*h);
	CI.setIdentity();
	VectorXd ci0(n*h);
	ci0.setZero();
	cout << "Inequality constraints built." << endl;
	
	// equality constraints, representing sum_i(w_ij) = 1
	MatrixXd CE(n*h,n);
	for(int i=0; i<h; i++) {
		CE.block(i*n,0,n,n).setIdentity();
	}
	VectorXd ce0(n);
	ce0.setConstant(-1);
	
	cout << "Equality constraints built." << endl;
	
	// solve
	VectorXd X;
	solve_quadprog(G,g0,CE,ce0,CI,ci0,X);
	cout << GREENGIN("Weight solved.") << endl;
	
	cout << CE.transpose()*X+ce0 << endl;
	
	// reshape new weights
	nW.resize(n,h);
	for(int i=0; i<h; i++) {
		nW.col(i) = X.segment(i*n,n);
	}
	cout << nW << endl;
}

void solve_weights_sparse(
	const std::vector<Pose> & poses, 
	const Eigen::MatrixXd & W,
	Eigen::MatrixXd & nW)
{
	
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
	Measure<>::execution(solve_weights_dense, poses, W, nW);
	double max_error = (nW-W).array().abs().maxCoeff();
	cout << "max error: " << max_error << endl;
	if( max_error > 1e-4) {
		cout << "Weight Diff:\n" << nW-W << endl;
	}
	
	return 0;
}