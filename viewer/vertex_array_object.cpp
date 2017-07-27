#include "vertex_array_object.h"

// #include <OpenGL/gl3.h>
// #include <OpenGL/gl3ext.h>

// we switch from glut to glfw
#include <GLFW/glfw3.h>

#include <Eigen/Core>
#include <iostream>
#include <string>
#include <cstdlib> // exit()
#include <algorithm> // find()
#include <cstdio>
#include <igl/sort.h>

#include "flattener.h"
#include "../pythonlike.h"
#include "shaderHelper.h"

// Attribute locations
#include "shaders/attributes.glsl"

// extern GLuint mesh_program;
void setup_render_mesh_VAO( const MeshType& mesh, GLuint VAO, const Eigen::MatrixXd & W )
{
	// Colors are per-vertex attributes.
	assert( 0 == mesh.FTC.rows() || mesh.F.rows() == mesh.FTC.rows() );
	
	// align uv coordinates and vertex positions for each triangle
	MeshVerticesType V_out;
	MeshUVType TC_out;
	MeshFacesType F_out;
	MeshVerticesType N_out;
	MeshColorsType C_out;
	
	flatten_attribute( mesh.F, mesh.V, V_out );
    flatten_attribute( mesh.FTC, mesh.TC, TC_out );
    flatten_attribute( mesh.FN, mesh.CN, N_out );
    flatten_attribute( mesh.F, mesh.C, C_out );
	flattened_face_indices( mesh.F.rows(), mesh.F.cols(), F_out );
		
	// generate buffers
	GLuint VBO; 
	GLuint FBO;
	GLuint NBO;
	GLuint CBO;
	GLuint TC_BO;
	glGenBuffers(1, &VBO);
    glGenBuffers(1, &FBO);
    glGenBuffers(1, &NBO);
    glGenBuffers(1, &CBO);
    glGenBuffers(1, &TC_BO);

	// bind vertex array object	
    glBindVertexArray(VAO);
    GLint id;

	// bind vertex positions to VAO
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER,
                 sizeof(float)*V_out.size(),
                 V_out.data(),
                 GL_STATIC_DRAW);
	
	// enable attribute for positions
	// GLuint program = mesh_program;
	// id = attrib(program, "position");
	id = POSITION_ATTRIB_LOCATION;
    glVertexAttribPointer(id,3,GL_FLOAT,GL_FALSE,0*sizeof(GLfloat),(GLvoid*)0);
    glEnableVertexAttribArray(id);

	// bind uv coordinates to VAO
    glBindBuffer(GL_ARRAY_BUFFER, TC_BO);
    glBufferData(GL_ARRAY_BUFFER,
                 sizeof(float)*TC_out.size(),
                 TC_out.data(),
                 GL_STATIC_DRAW);

    // enable attribute for uv coordiantes             
	// id = attrib(program, "vertexUV");
	id = TEXCOORD_ATTRIB_LOCATION;
	glVertexAttribPointer(id,2,GL_FLOAT,GL_FALSE,0*sizeof(GLfloat),(GLvoid*)0);
	glEnableVertexAttribArray(id);

	// bind normals to VAO
    glBindBuffer(GL_ARRAY_BUFFER, NBO);
    glBufferData(GL_ARRAY_BUFFER,
                 sizeof(float)*N_out.size(),
                 N_out.data(),
                 GL_STATIC_DRAW);

    // enable attribute for normals             
	// id = attrib(program, "normal");
	id = NORMAL_ATTRIB_LOCATION;
	glVertexAttribPointer(id,3,GL_FLOAT,GL_FALSE,0*sizeof(GLfloat),(GLvoid*)0);
	glEnableVertexAttribArray(id);
	
	// bind colors to VAO
    glBindBuffer(GL_ARRAY_BUFFER, CBO);
    glBufferData(GL_ARRAY_BUFFER,
                 sizeof(float)*C_out.size(),
                 C_out.data(),
                 GL_STATIC_DRAW);

    // enable attribute for colors           
	// id = attrib(program, "color");
	id = COLOR_ATTRIB_LOCATION;
	glVertexAttribPointer(id,4,GL_FLOAT,GL_FALSE,0*sizeof(GLfloat),(GLvoid*)0);
	glEnableVertexAttribArray(id);
	                
	// bind the indices to VAO
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, FBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 sizeof(GLuint)*F_out.size(),
                 F_out.data(),
                 GL_STATIC_DRAW);
    
    // release bondage             	
	glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    // delete buffers    
    glDeleteBuffers(1,&VBO);
    glDeleteBuffers(1,&FBO);
    glDeleteBuffers(1,&NBO); 
    glDeleteBuffers(1,&CBO); 
    glDeleteBuffers(1,&TC_BO); 
}
	
void setup_VAO_with_weights( 
	const MeshType& mesh, 
	GLuint VAO, 
	const MeshVerticesType & W,
	const MeshFacesType & IW )
{
	// Colors are per-vertex attributes.
	assert( 0 == mesh.FTC.rows() || mesh.F.rows() == mesh.FTC.rows() );
	
	// align uv coordinates and vertex positions for each triangle
	MeshVerticesType V_out;
	MeshUVType TC_out;
	MeshFacesType F_out;
	MeshVerticesType N_out;
	
	flatten_attribute( mesh.F, mesh.V, V_out );
    flatten_attribute( mesh.FTC, mesh.TC, TC_out );
    flatten_attribute( mesh.FN, mesh.CN, N_out );
	flattened_face_indices( mesh.F.rows(), mesh.F.cols(), F_out );

	// set mesh colors	
	MeshColorsType C_out(F_out.rows()*3, 4);
	C_out.setOnes();
	for( int i=0; i<F_out.rows(); i++ ) {
		C_out.row(i*3) << COLORS[i%10][0]/255., COLORS[i%10][1]/255., COLORS[i%10][2]/255., 1.;
		C_out.row(i*3+1) << COLORS[i%10][0]/255., COLORS[i%10][1]/255., COLORS[i%10][2]/255., 1.;
		C_out.row(i*3+2) << COLORS[i%10][0]/255., COLORS[i%10][1]/255., COLORS[i%10][2]/255., 1.;
	}
	
	MeshVerticesType W_out;
	MeshFacesType IW_out;
	flatten_attribute( mesh.F, W, W_out );
	flatten_attribute( mesh.F, IW, IW_out );
		
	// generate buffers
	GLuint VBO; 
	GLuint FBO;
	GLuint NBO;
	GLuint CBO;
	GLuint TC_BO;
	GLuint WBO;
	GLuint IWBO;
	glGenBuffers(1, &VBO);
    glGenBuffers(1, &FBO);
    glGenBuffers(1, &NBO);
    glGenBuffers(1, &CBO);
    glGenBuffers(1, &TC_BO);
    glGenBuffers(1, &WBO);
    glGenBuffers(1, &IWBO);

	// bind vertex array object	
    glBindVertexArray(VAO);
    GLint id;

	// bind vertex positions to VAO
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER,
                 sizeof(float)*V_out.size(),
                 V_out.data(),
                 GL_STATIC_DRAW);
	
	// enable attribute for positions
	// GLuint program = mesh_program;
	// id = attrib(program, "position");
	id = POSITION_ATTRIB_LOCATION;
    glVertexAttribPointer(id,3,GL_FLOAT,GL_FALSE,0*sizeof(GLfloat),(GLvoid*)0);
    glEnableVertexAttribArray(id);

	// bind uv coordinates to VAO
    glBindBuffer(GL_ARRAY_BUFFER, TC_BO);
    glBufferData(GL_ARRAY_BUFFER,
                 sizeof(float)*TC_out.size(),
                 TC_out.data(),
                 GL_STATIC_DRAW);

    // enable attribute for uv coordiantes             
	// id = attrib(program, "vertexUV");
	id = TEXCOORD_ATTRIB_LOCATION;
	glVertexAttribPointer(id,2,GL_FLOAT,GL_FALSE,0*sizeof(GLfloat),(GLvoid*)0);
	glEnableVertexAttribArray(id);

	// bind normals to VAO
    glBindBuffer(GL_ARRAY_BUFFER, NBO);
    glBufferData(GL_ARRAY_BUFFER,
                 sizeof(float)*N_out.size(),
                 N_out.data(),
                 GL_STATIC_DRAW);

    // enable attribute for normals             
	// id = attrib(program, "normal");
	id = NORMAL_ATTRIB_LOCATION;
	glVertexAttribPointer(id,3,GL_FLOAT,GL_FALSE,0*sizeof(GLfloat),(GLvoid*)0);
	glEnableVertexAttribArray(id);
	
	// bind colors to VAO
    glBindBuffer(GL_ARRAY_BUFFER, CBO);
    glBufferData(GL_ARRAY_BUFFER,
                 sizeof(float)*C_out.size(),
                 C_out.data(),
                 GL_STATIC_DRAW);

    // enable attribute for colors           
	// id = attrib(program, "color");
	id = COLOR_ATTRIB_LOCATION;
	glVertexAttribPointer(id,4,GL_FLOAT,GL_FALSE,0*sizeof(GLfloat),(GLvoid*)0);
	glEnableVertexAttribArray(id);
	
	// bind weight indices to VAO
	glBindBuffer(GL_ARRAY_BUFFER, IWBO);
    glBufferData(GL_ARRAY_BUFFER,
                 sizeof(unsigned int)*IW_out.size(),
                 IW_out.data(),
                 GL_STATIC_DRAW);

    // enable attribute for weight indices          
	// id = attrib(program, "weight_indices");
	id = WEIGHT_ATTRIB_LOCATION;
	glVertexAttribPointer(id,4,GL_UNSIGNED_INT,GL_FALSE,0*sizeof(GLuint),(GLvoid*)0);
	glEnableVertexAttribArray(id);
	
	// bind weights to VAO
	glBindBuffer(GL_ARRAY_BUFFER, WBO);
    glBufferData(GL_ARRAY_BUFFER,
                 sizeof(float)*W_out.size(),
                 W_out.data(),
                 GL_STATIC_DRAW);

    // enable attribute for weights           
	// id = attrib(program, "weights");
	id = WEIGHT_ATTRIB_LOCATION;
	glVertexAttribPointer(id,4,GL_FLOAT,GL_FALSE,0*sizeof(GLfloat),(GLvoid*)0);
	glEnableVertexAttribArray(id);
	                
	// bind the indices to VAO
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, FBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 sizeof(GLuint)*F_out.size(),
                 F_out.data(),
                 GL_STATIC_DRAW);
    
    // release bondage             	
	glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    // delete buffers    
    glDeleteBuffers(1,&VBO);
    glDeleteBuffers(1,&FBO);
    glDeleteBuffers(1,&NBO); 
    glDeleteBuffers(1,&CBO); 
    glDeleteBuffers(1,&TC_BO); 
    glDeleteBuffers(1,&WBO); 
    glDeleteBuffers(1,&IWBO); 
}	

void reduce_weights_K_largest(
	const Eigen::MatrixXd & W, 
	MeshVerticesType & W_out, 
	MeshFacesType & IW_out,
	int weight_bit)
{
	using namespace std;
	using namespace Eigen;

	const int m = W.rows();
	const int n = W.cols();
	// reduce the size of weights to a fixed size
	
	W_out.resize(m, weight_bit);
	IW_out.resize(m, weight_bit);
	if(n <= weight_bit) {
		for(int i=0; i<weight_bit; i++) {
			if(i<n)	W_out.col(i) = W.col(i).cast<float>();
			else	W_out.col(i).setZero();
			IW_out.col(i).setConstant((unsigned int)(i));
		}
	}
	else {
		MatrixXd sorted_W(m, n);
		MatrixXi sorted_IW(m, n);
		igl::sort(W, 2, false, sorted_W, sorted_IW);
		for(int i=0; i<weight_bit; i++) {
			W_out.col(i) = sorted_W.col(i).cast<float>();
			IW_out.col(i) = sorted_IW.col(i).cast<unsigned int>();
		}
	}
}
	
GLuint general_pipeline_setup_mesh( const MeshType & mesh, const Eigen::MatrixXd & W )
{
	using namespace std;
	using namespace Eigen;
	
	GLuint VAO;
	glGenVertexArrays(1, &VAO);
	
	setup_render_mesh_VAO(mesh, VAO, W);
// 	MeshVerticesType W_out;
// 	MeshFacesType IW_out;
// 	reduce_weights_K_largest(W, W_out, IW_out, 4);
// 	setup_VAO_with_weights( mesh, VAO, W_out, IW_out );
	return VAO;
}

