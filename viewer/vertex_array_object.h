#ifndef __vertex_array_object_h__
#define __vertex_array_object_h__

#include <string>
#include "glcompat.h"
#include "common.h"
#include <igl/Camera.h>
#include <Eigen/Core>

// Doesn't pass a program because the layout is defined in "../shaders/attributes.glsl"
void setup_render_mesh_VAO( 
	const MeshType& mesh,
	GLuint VAO,
	const Eigen::MatrixXd & W );

void setup_VAO_with_weights( 
	const MeshType& mesh, 
	GLuint VAO, 
	const Eigen::MatrixXd & W,
	const Eigen::MatrixXi & IW );
	
// setup mesh rendering.
// Returns a new glGenVertexArrays(). Cleanup with glDeleteVertexArrays().
GLuint general_pipeline_setup_mesh( const MeshType & mesh, const Eigen::MatrixXd & W );

		
#endif
