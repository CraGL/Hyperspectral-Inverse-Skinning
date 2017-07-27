#include "flattener.h"

void flatten_attributes(
	// Input
	const MeshVerticesType& V, const MeshFacesType& FV,
	const MeshUVType& TC, const MeshFacesType& FTC,
	const MeshVerticesType& N, const MeshFacesType& FN,
	// Output
	MeshVerticesType& V_out, MeshUVType& TC_out, MeshVerticesType& N_out
	)
{
	assert( FV.rows() == FTC.rows() );
	assert( FV.rows() == FN.rows() );
	assert( FV.cols() == FTC.cols() );
	assert( FV.cols() == FN.cols() );

    flatten_attribute( FV, V, V_out );
    flatten_attribute( FTC, TC, TC_out );
    flatten_attribute( FN, N, N_out );
}
	
void flattened_face_indices( const int F_rows, const int F_cols, MeshFacesType& F_out )
{
	F_out.resize( F_rows, F_cols );
	int count = 0;
	for( int row = 0; row < F_rows; ++row ) {
		for( int col = 0; col < F_cols; ++col ) {
			F_out( row, col ) = count;
			count += 1;
		}
	}
}