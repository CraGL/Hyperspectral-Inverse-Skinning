#include <igl/readDMAT.h>
#include <igl/writeDMAT.h>
#include <igl/REDRUM.h>

#include <iostream>
#include <fstream>

#include <Eigen/Core>
#include "pythonlike.h"
#include <vector>

namespace {
	void usage( const char* argv0 ) {
		std::cerr<<"Usage:"<<std::endl<<"    " << argv0 << " weight_path.DMAT index1 fixed_weight1 ... "<< std::endl;
		// exit(0) means success. Anything else means failure.
		exit(-1);
    }
}

int main(int argc, char * argv[])
{
	using namespace std;
	using namespace Eigen;
	
	if( argc <= 2 || argc % 2 != 0 ) {
		usage( argv[0] );
	}
	
	string weight_path( argv[1] );
	vector<string> args( argv + 2, argv + argc );
	
	MatrixXd W;
	if(!igl::readDMAT(weight_path, W)) {
		cerr << REDRUM("Reading weights fails.");
		exit(-1);
	}
	
	vector<bool> unfixed( W.cols(), true );
	
// 	cout << "Before touch: " << endl;
// 	cout << W << endl;
	
	for( int i=0; i<args.size(); i+=2 ) {
		int index = stoi( args[i] );
		double w = stod( args[i+1] );
		
		unfixed[index] = false;
		
		double max_w = W.col(index).maxCoeff();
		W.col(index) = W.col(index)*w/max_w;
	}
	
	for( int i=0; i<W.cols(); i++ )
		cout << W.col(i).maxCoeff() << " ";
	cout << endl;
	
	for( int i=0; i<W.rows(); i++ ) {
		double sum_fixed = 0;
		for( int j=0; j<W.cols(); j++ ) {
			if( unfixed[j] == false )
				sum_fixed += W(i,j);
		}
		
		if( W.row(i).sum() != sum_fixed ) {
			double ratio = ( 1-sum_fixed)/(W.row(i).sum() - sum_fixed );
			for( int j=0; j<W.cols(); j++ ) {
				if( unfixed[j] == true )
					W(i,j) *= ratio;
			}
		}
		else
			W.row(i) /=  W.row(i).sum();
	}
	
// 	cout << "After touch: " << endl;
// 	cout << W << endl;
	
	// save weights to file
	string outpath_base = pythonlike::os_path_splitext( weight_path ).first;
	string outpath = outpath_base + "_dirty.DMAT";
	igl::writeDMAT(outpath, W);
}