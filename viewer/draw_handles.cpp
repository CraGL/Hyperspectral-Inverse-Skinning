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
#include <igl/REDRUM.h>

extern std::string shader_path;

MeshType sphere_mesh, arrow_mesh;
GLuint handle_program;
GLuint s_VAO, x_VAO, y_VAO, z_VAO;
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
	GLuint VAO,
	const Eigen::RowVector4f & color)
{	
	GLuint VBO;
	GLuint CBO;
	GLuint EBO; 
	glGenBuffers(1, &VBO);
	glGenBuffers(1, &CBO);
    glGenBuffers(1, &EBO);
	
	MeshColorsType C(V.rows(), 4);
	for( int i=0; i<V.rows(); i++ ) {
		C.row(i) << color(0), color(1), color(2), color(3);
// 		C.row(i*3+1) << color(0), color(1), color(2), color(3);
// 		C.row(i*3+2) << color(0), color(1), color(2), color(3);
	}
	
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER,
                 sizeof(float)*V.size(),
                 V.data(),
                 GL_STATIC_DRAW);
	
	GLint id = attrib(handle_program, "position");
    glVertexAttribPointer(id,V.cols(),GL_FLOAT,GL_FALSE,0*sizeof(GLfloat),(GLvoid*)0);
    glEnableVertexAttribArray(id);
    
    // bind colors to VAO
    glBindBuffer(GL_ARRAY_BUFFER, CBO);
    glBufferData(GL_ARRAY_BUFFER,
                 sizeof(float)*C.size(),
                 C.data(),
                 GL_STATIC_DRAW);

	id = COLOR_ATTRIB_LOCATION;
	glVertexAttribPointer(id,C.cols(),GL_FLOAT,GL_FALSE,0*sizeof(GLfloat),(GLvoid*)0);
	glEnableVertexAttribArray(id);
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 sizeof(GLuint)*F.size(),
                 F.data(),
                 GL_STATIC_DRAW);
    
	glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    glDeleteBuffers(1,&VBO);
    glDeleteBuffers(1,&CBO);
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
	arrow_mesh.read_triangle_mesh("../viewer/arrow.obj");

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
	prepare_vertex_array_object(sphere_mesh.V, sphere_mesh.F, s_VAO, igl::MAYA_YELLOW);
	
	glGenVertexArrays(1, &x_VAO);
	prepare_vertex_array_object(arrow_mesh.V, arrow_mesh.F, x_VAO, igl::MAYA_RED);
	
	glGenVertexArrays(1, &y_VAO);
	prepare_vertex_array_object(arrow_mesh.V, arrow_mesh.F, y_VAO, igl::MAYA_GREEN);
	
	glGenVertexArrays(1, &z_VAO);
	prepare_vertex_array_object(arrow_mesh.V, arrow_mesh.F, z_VAO, igl::MAYA_BLUE);
	
	mappings.clear();
	for(int c = 0;c < C.rows();c++)
	{
		Matrix4d M;
		M.setIdentity();
		M.block(0,3,3,1) = C.row(c).transpose();
		double r = 0.02*half_bbd;
		Affine3d s2(Scaling(r,r,r));
 		mappings.push_back( M*s2.matrix() );
 		Affine3d t1(AngleAxisd(M_PI/2,Vector3d::UnitZ()));
 		mappings.push_back( M*t1.matrix()*s2.matrix() );
 		mappings.push_back( M*s2.matrix() );
 		Affine3d t2(AngleAxisd(M_PI/2,Vector3d::UnitX()));
 		mappings.push_back( M*t2.matrix()*s2.matrix() );
	}
}

void end_handles_3d()
{
	glDeleteVertexArrays(1, &s_VAO);
	glDeleteVertexArrays(1, &x_VAO);
	glDeleteVertexArrays(1, &y_VAO);
	glDeleteVertexArrays(1, &z_VAO);
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
		return draw_handles_3d(m_camera,C,T,igl::MAYA_YELLOW);
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
	
	auto draw_arrow = []( const igl::Camera & m_camera, GLuint vao, Eigen::Matrix4d M = Eigen::Matrix4d::Identity())
	{	
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		bind_projection(m_camera,M);	
		// Draw mesh as wireframe
		glBindVertexArray(vao);
		glDrawElements(GL_TRIANGLES, arrow_mesh.F.size(), GL_UNSIGNED_INT, 0);
		glBindVertexArray(0);
	};
	// Loop over bones
	for(int e=0;e<C.rows();e++)
	{
		auto bind_color = [e](const MatrixXf& color)
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
		
		Matrix4d Te = Matrix4d::Identity();
// 		std::cout << T.rows() << " " << T.cols() << " " << C.rows() << std::endl;
		Te.block(0,0,3,4) = T.block(e*4,0,4,3).transpose();
		bind_color(igl::MAYA_RED);
		draw_arrow( m_camera, x_VAO, Te*mappings[4*e+1] );
		bind_color(igl::MAYA_GREEN);
		draw_arrow( m_camera, y_VAO, Te*mappings[4*e+2] );
		bind_color(igl::MAYA_BLUE);
		draw_arrow( m_camera, z_VAO,Te*mappings[4*e+3] );
		bind_color(igl::MAYA_YELLOW);
		draw_sphere( m_camera,Te*mappings[4*e] );
	}
	glUseProgram( 0 );	
}