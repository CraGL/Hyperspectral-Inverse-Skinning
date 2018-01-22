#include "blend_scene.h"

#include <igl/REDRUM.h>
#include <igl/forward_kinematics.h>
#include <igl/readDMAT.h>
#include <igl/readTGF.h>
#include <igl/pathinfo.h>
#include <igl/bone_parents.h>
#include <igl/colon.h>
#include <igl/material_colors.h>
#include <igl/bounding_box.h>
#include <igl/list_to_matrix.h>

#include "../json/json.h"
#include "flattener.h"
#include "../pythonlike.h"
#include "vertex_array_object.h"


std::string shader_path = "../viewer/shaders/";

std::string customize_defines(const ShaderSettings & shader)
{ 
	std::string s("#version 410 core\n");
	return s +
			(shader.enable_normalize ? "#define NORMALIZE 1\n" : "") +
			"#define M_PI 3.1415926535897932384626433832795\n"
			;
}

void BlendScene::init_scene(
	const std::vector< std::string > & mesh_paths,
	const std::string & weight_path)
{
	using namespace std;
	using namespace Eigen;
	using namespace pythonlike;
	
	// Read input mesh from file
	P = mesh_paths.size();
	poses.reserve(P);
	
	if(!igl::readDMAT(weight_path, W)) 	cerr << REDRUM("Reading weights fails.");
	else {
		cout << GREENGIN("Reading weights succeeds.");
		H = W.cols();
	}
	
	poses.clear();	
	for(int i=0; i<P; i++) {
		MeshType pos;
		pos.read_triangle_mesh(mesh_paths[i]);
		if( pos.CN.size() == 0 ) {
			std::cout << "No normals in the mesh; computing per-vertex normals automatically." << std::endl;
			pos.compute_per_vertex_normals();
			pos.color_faces_with_weights(W);
		}
		poses.push_back(pos);
	}
	
	dQ.clear();
	T.resize(4*H, 3);
	T.setZero();
	for(int i=0; i<H; i++) {
		dQ.push_back(Eigen::Quaterniond(1,0,0,0));
		T.block(4*i,0,3,3).setIdentity();
	}
}

void BlendScene::load_results(std::string result_path)
{
	using namespace std;
	using namespace Eigen;
	using namespace pythonlike;
	
	string section = "";
	int count=0;
	int B=0;
	int nframes=0;
	int rows=0;
	vector< Matrix4d > all_transformations;
	
	string line;
	ifstream file (result_path);
	if (file.is_open()) {
		while ( getline (file,line) ) {
			vector<string> words = split( strip(line), ',' );
			
			if(words.size() == 3 && words[0] == "*BONEANIMATION") {
				section = "bone";
				nframes = stoi(words[2].substr(8));
				count=0;
				B += 1;
			}
			else if( words.size() > 0 and words[0] == "*VERTEXWEIGHTS" ) {
				section = "weight";
				rows = stoi( split( split(words[1], ' ')[0], '=' )[1] );
				W.resize(rows, B);
				count = 0;
			}
			else if( section == "bone" && count < nframes ) {
				vector<string> words = split(strip(line), ' ');
				assert( words.size() == 17 );
				Matrix4d M;
				for(int i=0; i<4; i++)
					for(int j=0; j<4; j++)
						M(i,j) = stod(words[1+i*4+j]);
				all_transformations.push_back( M );
				count += 1;
			}
			else if( section == "weight" && count < rows ) {
				vector<string> words = split(strip(line), ' ');
				assert( words.size() % 2 == 0 );
				int ridx = stoi( words[0] );
				int num = words.size()/2-1;
				for( int i=0; i<num; i++) {
					int cidx = stoi( words[i*2+2] );
					double val = stod( words[i*2+3] );
					W(ridx, cidx) = val;
				}
				count += 1;
			}
		}
		file.close();
	}

	else cout << "Unable to open file" << endl;
	
	assert( all_transformations.size() == B*nframes );
	P = nframes;
	H = B;
	for( int i=0; i<nframes; i++ ) {
		vector< Matrix4d > transformations_per_pose;
		for( int j=0; j<B; j++ ) {
			transformations_per_pose.push_back( all_transformations[nframes*j+i] );
		}
		bone_transformations.push_back( transformations_per_pose );
	}	
}

void BlendScene::init_scene(
	const std::string & rest_path, 
	const std::string & result_path)
{
	using namespace std;
	using namespace Eigen;
	using namespace pythonlike;
	
	load_results( result_path );
	// insert rest pose and its bone transformations
	P += 1;
	vector< Matrix4d > rest_transformations(H, Matrix4d::Identity());
	bone_transformations.insert(bone_transformations.begin(), rest_transformations);
	poses.reserve(P);	
	poses.clear();	
	MeshType rest_pose;
	rest_pose.read_triangle_mesh(rest_path);
	if( rest_pose.CN.size() == 0 ) {
		std::cout << "No normals in the mesh; computing per-vertex normals automatically." << std::endl;
		rest_pose.compute_per_vertex_normals();
		rest_pose.color_faces_with_weights(W);
	}
	poses.push_back(rest_pose);
	
	MatrixXd V = rest_pose.V.cast<double>();
	for(int k=1; k<P; k++) {
		MeshType pos;
		pos.copy( rest_pose );
		
		for( int i=0; i<V.rows(); i++ ) {
			Vector4d p = V.row(i).homogeneous().transpose();
			RowVector4d deformed_p = RowVector4d::Zero();
			for( int j=0; j<H; j++ ) {
				Matrix4d transform = bone_transformations[k][j];
				deformed_p += (transform*p).transpose()*W(i,j);
			}
			pos.V.row(i) = deformed_p.head(3).cast<float>();
		}
		pos.compute_per_vertex_normals();
		poses.push_back(pos);
	}
	
	dQ.clear();
	T.resize(4*H, 3);
	T.setZero();
	for(int i=0; i<H; i++) {
		dQ.push_back(Eigen::Quaterniond(1,0,0,0));
		T.block(4*i,0,3,3).setIdentity();
	}
}

void BlendScene::show_pose(const int index_pos) {
	
	using namespace std;
	using namespace Eigen;
	
	assert( index_pos < P && index_pos >= 0);
	MeshType curr_pos = poses[index_pos];
	
	// set camera
	const RowVector3f Vmin = curr_pos.V.colwise().minCoeff();
	const RowVector3f Vmax = curr_pos.V.colwise().maxCoeff();
	const Eigen::Vector3f Vmid = curr_pos.V.colwise().mean().transpose();
	bbd = (Vmax-Vmin).norm(); //(Vmax-Vmin).maxCoeff();

	const Vector3d cen = Vmid.cast<double>();
	// set camera
	camera.look_at(
		cen+Eigen::Vector3d(0,0,2.*bbd),
		cen,
		Eigen::Vector3d(0,1,0));
	camera.m_far = 500;
	
	pose_index = index_pos;
	
}

void BlendScene::compute_handle_positions()
{	
	using namespace std;
	using namespace Eigen;
	
	MatrixXd V = poses[pose_index].V.transpose().cast<double>();
	C = (V*W).transpose();
	for(int i=0; i<H; i++) {
		C.row(i) /= W.col(i).sum();
	}
}

void BlendScene::bind_scene_varying_uniforms(GLuint program_id) const
{
	using namespace std;
	using namespace Eigen;

	GLint location;
	glUseProgram(program_id);

	// set view * proj uniform
	Matrix4f proj = camera.projection().cast<float>();
	Matrix4f view = camera.inverse().matrix().cast<float>();
	Matrix4f mvp = proj*view;
	location = uniform(program_id,"modelViewProj");
	glUniformMatrix4fv(location,1,GL_FALSE,mvp.data());

	// set viewport uniform	
	location = uniform(program_id,"eye_pos");
	Eigen::Vector3f eye = camera.eye().cast<float>();
	glUniform3fv(location,1, eye.data());

	location = uniform(program_id,"selected_handle");
	glUniform1i( location, selected_handle );
	// set colorstyle uniform
	const bool enable_color_weight = 
		( shader_settings.color_style == COLOR_WEIGHT ) ? true : false;
	location = uniform(program_id,"enable_color_weight");
	glUniform1i(location, enable_color_weight);

}

/// bind uniform threshold and transformation to shader program. 
void BlendScene::bind_transforms( 
	GLuint program_id, 
	bool is_animated)
{
	using namespace std;
	using namespace Eigen;
	glUseProgram(program_id);
	
	if( selected_handle != -1 ) {
		dQ[selected_handle] = rotation;
	}

	// set transformation
	for( int i=0; i<H; i++ ) {
		std::string name = "transforms[" + std::to_string(i) + "]";
		GLint location = uniform(program_id, name.c_str());
		Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
	
		transform.block(0,0,3,4) = T.block(i*4,0,4,3).transpose().template cast<float>();
		glUniformMatrix4fv(location,1,GL_FALSE,transform.data());
	}
}

void BlendScene::build_mesh_program(GLuint &program_id)
{
	using namespace std;
	
	std::vector< GLuint > shaders;
	
	cout << "Render mesh without Tessellation." << endl;
	shaders.push_back( init_vertex_shader(program_id, 
		std::vector<std::string>{
			customize_defines(shader_settings),
			file_to_string(shader_path + "attributes.glsl"),
			file_to_string(shader_path + "noTessTop.vs"),
// 			file_to_string(shader_path + "deform.glsl"),
			file_to_string(shader_path + "noTessBottom.vs"),
		}) );
	shaders.push_back( init_fragment_shader(program_id, file_to_string(shader_path + "meshPlain.fs")) ); 
	
	for( const auto& shader : shaders ) {
	    if( !shader ) {
    		cerr << "shader initialization fails" << endl;
	    	exit(-1);
	    }
	}
	
    if( !link_shaders(program_id) ) {
		cerr << "shader linkage fails" << endl;
		exit(-1);
	}
	
	// Free the shaders.
	for( const auto& shader : shaders ) {
	    glDetachShader( program_id, shader );
	    glDeleteShader( shader );
	}
}


void BlendScene::build_wire_program(GLuint &program_id)
{
	using namespace std;
	
	std::vector< GLuint > shaders;
	
	cout << "Render wire without Tessellation." << endl;
	shaders.push_back( init_vertex_shader(program_id, 
		std::vector<std::string>{
			customize_defines(shader_settings),
			file_to_string(shader_path + "attributes.glsl"),
			file_to_string(shader_path + "noTessTop.vs"),
// 			file_to_string(shader_path + "deform.glsl"),
			file_to_string(shader_path + "noTessBottom.vs"),
		}) );
	shaders.push_back( init_fragment_shader(program_id, file_to_string(shader_path + "wirePlain.fs")) );
	
    if( !link_shaders(program_id) ) {
		cerr << "shader linkage fails" << endl;
		exit(-1);
	}
	
	// Free the shaders.
	for( const auto& shader : shaders ) {
	    glDetachShader( program_id, shader );
	    glDeleteShader( shader );
	}
}
