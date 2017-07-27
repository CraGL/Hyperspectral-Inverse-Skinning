#ifndef __GPU_COMMON__
#define __GPU_COMMON__

#include <Eigen/Core>
#include <Eigen/StdVector> // weird: https://github.com/libigl/libigl/issues/412
#include <iostream>

#include <igl/list_to_matrix.h>
#include <igl/polygon_mesh_to_triangle_mesh.h>
#include <igl/readOBJ.h>
#include <igl/per_vertex_normals.h>

typedef std::vector<
Eigen::Quaterniond,
  Eigen::aligned_allocator<Eigen::Quaterniond> > RotationList;

// Mesh data: RowMajor is important to directly use in OpenGL
typedef Eigen::Matrix< float,Eigen::Dynamic,3,Eigen::RowMajor> MeshVerticesType;
typedef Eigen::Matrix< unsigned int,Eigen::Dynamic,3,Eigen::RowMajor> MeshFacesType;
typedef Eigen::Matrix< float,Eigen::Dynamic,2,Eigen::RowMajor> MeshUVType;
typedef Eigen::Matrix< float,Eigen::Dynamic,4,Eigen::RowMajor> MeshColorsType;

// colors come from http://tools.medialab.sciences-po.fr/iwanthue/
const float COLORS[50][3] =
{
	{174,214,174},
	{164,65,228},
	{123,221,61},
	{80,77,222},
	{210,227,66},
	{83,27,159},
	{91,217,111},
	{198,62,201},
	{106,157,52},
	{107,75,187},
	{203,183,62},
	{99,121,229},
	{223,154,51},
	{60,27,107},
	{168,225,131},
	{210,55,159},
	{100,226,170},
	{187,113,225},
	{76,159,100},
	{218,64,110},
	{69,221,206},
	{223,70,51},
	{83,194,223},
	{147,54,40},
	{132,210,205},
	{126,52,126},
	{208,210,132},
	{62,70,136},
	{202,111,59},
	{86,130,202},
	{135,115,48},
	{214,108,180},
	{61,96,37},
	{221,157,215},
	{48,60,39},
	{154,123,194},
	{105,138,103},
	{138,48,82},
	{213,209,191},
	{52,29,55},
	{218,169,129},
	{62,87,114},
	{219,128,137},
	{78,133,132},
	{89,45,32},
	{165,193,211},
	{129,93,117},
	{120,160,207},
	{149,115,97},
	{205,175,202}
};


struct MeshType
{	
	// Vertices
	MeshVerticesType V;
	// Faces
	MeshFacesType F;
	// Per-vertex uv
	MeshUVType TC;
	// uv indices per face
	MeshFacesType FTC;
	// Per-vertex normal
	MeshVerticesType CN;
	// normal indices per face
	MeshFacesType FN;
	// color used by CPU test
	MeshColorsType C;
	
	void copy( MeshType target ) {
		this->V = target.V;
		this->F = target.F;
		this->TC = target.TC;
		this->FTC = target.FTC;
		this->CN = target.CN;
		this->FN = target.FN;
		this->C = target.C;
	}
	
	bool read_triangle_mesh(const std::string str)
	{
		using namespace std;
		using namespace Eigen;
		using namespace igl;
	
		vector<vector<double > > vV,vN,vTC;
		vector<vector<int > > vF,vFTC,vFN;
	
		if(!readOBJ(str,vV,vTC,vN,vF,vFTC,vFN))
		{
			return false;
		}
	
		if(vV.size() > 0)
		{
			if(!list_to_matrix(vV,this->V))
			{
			  return false;
			}
			polygon_mesh_to_triangle_mesh(vF,this->F);
		}
		if(vTC.size() > 0)
		{
		    // Sometimes 3D texture coordinates come in.
		    // Drop the last coordinate before calling list_to_matrix().
		    for( auto& tc : vTC ) {
		        assert( tc.size() == 2 || tc.size() == 3 );
		        tc.resize(2);
		    }
		    
			if(!list_to_matrix(vTC,this->TC))
			{
			  return false;
			}
			polygon_mesh_to_triangle_mesh(vFTC,this->FTC);
		}
		if(vN.size() > 0)
		{
			if(!list_to_matrix(vN,this->CN))
			{
			  return false;
			}
			polygon_mesh_to_triangle_mesh(vFN,this->FN);
		}
		return true;
	}
	
	void compute_per_vertex_normals()
	{
		// Computer per-vertex normals.
		igl::per_vertex_normals( this->V, this->F, this->CN );
		// Now CN has the same indexing structure as V.
		this->FN = this->F;
	}
	
	void color_faces_with_weights(const Eigen::MatrixXd & W) {
		std::vector<unsigned int> indices;
		for(int i=0; i<W.cols(); i++) {
			indices.push_back(i);
		}
		color_faces_with_weights(W, indices);
	}
	
	void color_faces_with_weights(const Eigen::MatrixXd & W, 
		std::vector<unsigned int> & indices ) {
		assert(W.rows() == V.rows());
		assert(W.cols() < 50);
		assert(indices.size() == W.cols());
		C.resize(W.rows(), 4);
		
		Eigen::MatrixXd handle_colors(W.cols(), 4);
		for(int i=0; i<W.cols(); i++) {
			handle_colors.row(i) << COLORS[indices[i]][0]/255.,COLORS[indices[i]][1]/255.,COLORS[indices[i]][2]/255.,1.;
		}
		
		Eigen::MatrixXf colors = (W*handle_colors).cast<float>();
		for(int i=0; i<colors.rows(); i++){
			C.row(i) = colors.row(i);
		}
	}
};

#include "shaders/attributes.glsl"

#endif
