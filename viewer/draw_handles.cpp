#include "glcompat.h"
// #include <OpenGL/gl3.h>
// #include <OpenGL/gl3ext.h>

#include <GLFW/glfw3.h>

#include <igl/PI.h>
#include <igl/material_colors.h>
#include <Eigen/Geometry>
#include <iostream>
#include "common.h"
#include "shaderHelper.h"
#include "draw_handles.h"

extern std::string shader_path;

MeshType sphere_mesh;
GLuint handle_program;
GLuint s_VAO;
std::vector<Eigen::Matrix4d> mappings;

void bind_projection( const igl::Camera & m_camera, const Eigen::Matrix4d & M)
{
	using namespace Eigen;
	
	glUseProgram(handle_program);
	Matrix4f proj = m_camera.projection().cast<float>();
	Matrix4f view = m_camera.inverse().matrix().cast<float>();
	Matrix4f mvp = proj*view*(M.cast<float>());
	
	GLint modelViewProj_loc = uniform(handle_program,"modelViewProj");
	glUniformMatrix4fv(modelViewProj_loc,1,GL_FALSE,mvp.data());
}

GLuint prepare_vertex_array_object( 
	const MeshVerticesType V, 
	const MeshFacesType F,
	GLuint VAO)
{	
	GLuint VBO;
	GLuint EBO; 
	glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);
	
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER,
                 sizeof(float)*V.size(),
                 V.data(),
                 GL_STATIC_DRAW);
	
	GLint id = attrib(handle_program, "position");
    glVertexAttribPointer(id,V.cols(),GL_FLOAT,GL_FALSE,0*sizeof(GLfloat),(GLvoid*)0);
    glEnableVertexAttribArray(id);
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 sizeof(GLuint)*F.size(),
                 F.data(),
                 GL_STATIC_DRAW);
    
	glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    glDeleteBuffers(1,&VBO);
    glDeleteBuffers(1,&EBO); 
    
    return VAO;
}

void init_handles_3d(
	const Eigen::MatrixXd & C,
	const double half_bbd)
{
	using namespace std;
	using namespace Eigen;
	
    assert(C.cols() == 3);
	sphere_mesh.read_triangle_mesh("../viewer/sphere.obj");

	const std::string handle_vertex_shader =  
		R"(
		#version 410 core

		uniform mat4 modelViewProj;
		in vec3 position;

		void main()
		{
			gl_Position = modelViewProj * vec4(position,1.);
		}
		)"; 
	const std::string handle_fragment_shader = 
		R"(
		#version 410 core

		uniform vec4 color;
		out vec4 frag_color;

		void main()
		{
			frag_color = color;
		}
		)";

	bool status = build_shaders_from_strings(handle_program, 
		handle_vertex_shader,
		handle_fragment_shader);
		
	if( status != true ) {
		cout << "init and link skeleton shaders fails. Exit." << endl;
		exit(-1);		
	}
	glUseProgram( handle_program );
	
	glGenVertexArrays(1, &s_VAO);
	prepare_vertex_array_object(sphere_mesh.V, sphere_mesh.F, s_VAO);
	mappings.clear();
	for(int c = 0;c < C.rows();c++)
	{
		Matrix4d M;
		M.setIdentity();
		M.block(0,3,3,1) = C.row(c).transpose();
		double r = 0.02*half_bbd;
		Affine3d s2(Scaling(r,r,r));
 		mappings.push_back( M*s2.matrix() );
	}
}

void end_handles_3d()
{
	glDeleteVertexArrays(1, &s_VAO);
	glDeleteProgram( handle_program );
}

void draw_handles_3d(
	const igl::Camera & m_camera, 
	const Eigen::MatrixXd & C,
	const Eigen::MatrixXd & T,
	const Eigen::MatrixXf & color)
{
	// Note: Maya's skeleton *does* scale with the mesh suggesting a scale
	// parameter. Further, its joint balls are not rotated with the bones.
	using namespace Eigen;
	using namespace std;
	assert(C.cols() == 3);
	if(color.size() == 0)
	{
		return draw_handles_3d(m_camera,C,T,igl::MAYA_PURPLE);
	}
	assert(color.cols() == 4 || color.size() == 4);
	glUseProgram( handle_program );		
	auto draw_sphere = []( const igl::Camera & m_camera, Eigen::Matrix4d M = Eigen::Matrix4d::Identity())
	{
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		bind_projection(m_camera,M);
		// Draw mesh as wireframe
		glBindVertexArray(s_VAO);
		glDrawElements(GL_TRIANGLES, sphere_mesh.F.size(), GL_UNSIGNED_INT, 0);
		glBindVertexArray(0);
	};
	// Loop over bones
	for(int e=0;e<C.rows();e++)
	{
		auto bind_color = [e, &color]()
		{
			glUseProgram(handle_program);
			GLint color_loc = uniform(handle_program,"color");
			if(color.size() == 4)
			{
				glUniform4f( color_loc, color(0), color(1), color(2), color(3));
			}else
			{
				glUniform4f( color_loc, color(e,0), color(e,1), color(e,2), color(e,3));
			}
		};
		
		bind_color();
		
		Matrix4d Te = Matrix4d::Identity();
// 		std::cout << T.rows() << " " << T.cols() << " " << C.rows() << std::endl;
		Te.block(0,0,3,4) = T.block(e*4,0,4,3).transpose();
		draw_sphere( m_camera,Te*mappings[e] );
	}
	glUseProgram( 0 );	
}