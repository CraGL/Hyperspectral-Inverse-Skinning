#ifndef __FLATTENER__
#define __FLATTENER__

#include "common.h" // For: Eigen::MatrixXf, MeshFacesType, MeshUVType
/*
Given an #faces-by-3 matrix of faces `F`, where each row of `F` contains the
three indices into `attribute` for the three vertices of `F`,
and an N-by-M matrix of `attributes`,
upon return fills the output variable `attribute_out` such that
each row is the attribute corresponding to the flattened vertices of `F`.
The flattened data order is [ face0_vertex0, face0_vertex1, face0_vertex2, face1_vertex0, ... ].
*/
template< typename T >
void flatten_attribute(
	// Input
	const MeshFacesType& F, const T& attribute,
	// Output
	T& attribute_out
	)
{
	// 3 vertices per triangle.
	assert( F.cols() == 3 );
	
	// The output attribute has the same number of columns as the input attribute.
	// It has 3 times the number of faces.
	attribute_out.resize( 3*F.rows(), attribute.cols() );
	
	// Iterate over every vertex of every triangle.
	// Output the attributes for each one.
	int count = 0;
	for( int face_index = 0; face_index < F.rows(); ++face_index ) {
		for( int face_vertex_index = 0; face_vertex_index < F.cols(); ++face_vertex_index ) {
			// The vertex for this triangle coordinate is:
			const int attribute_index = F( face_index, face_vertex_index );
			
			// Push the attribute to the end of the output (attribute_out).
			attribute_out.row( count ) = attribute.row( attribute_index );
			
			count += 1;
		}
	}
}

void flatten_attributes(
	// Input
	const MeshVerticesType& V, const MeshFacesType& FV,
	const MeshUVType& TC, const MeshFacesType& FTC,
	const MeshVerticesType& N, const MeshFacesType& FN,
	// Output
	MeshVerticesType& V_out, MeshUVType& TC_out, MeshVerticesType& N_out
	);
	
void flattened_face_indices( const int F_rows, const int F_cols, MeshFacesType& F_out );

#endif
