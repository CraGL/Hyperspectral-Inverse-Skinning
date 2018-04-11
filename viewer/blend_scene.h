#ifndef __blend_scene_h__
#define __blend_scene_h__

#include <igl/Camera.h>
#include "common.h"
#include "shaderHelper.h"

enum ColorStyleType
{
	NO_COLOR = 0,
	COLOR_WEIGHT,
	COLOR_TYPE_NUM
};


// Settings which determine which shader programs to load.
// Not for parameters read by shaders.
struct ShaderSettings
{
	bool enable_normalize;
    ColorStyleType color_style;
	
	ShaderSettings() :
		enable_normalize( true ),
		color_style( COLOR_WEIGHT )
	{}
	
	bool operator==( const ShaderSettings& rhs ) const
	{
		return
			enable_normalize == rhs.enable_normalize &&
			color_style == rhs.color_style
			;
	}
	
	bool operator!=( const ShaderSettings& rhs ) const
	{
		return !( (*this) == rhs );
	}
};

struct BlendScene
{    
	// same as static scene
	std::vector< MeshType > poses;
	int pose_index;
    
    Eigen::Vector4i viewPort = Eigen::Vector4i::Zero();
    igl::Camera camera;
    Eigen::Vector3f eye;
    
    std::string scene_path;
    
    double bbd;	 // bounding box diagonal
    int P;		// number of poses;
    int H; 		// number of handles;
    
    Eigen::MatrixXd T;		// transformation matrices
    Eigen::MatrixXd W;		// weights
    Eigen::MatrixXd C; 		// estimated controls
    
    std::vector< std::vector< Eigen::Matrix4d > > bone_transformations;	// Pose-by-bone-by-homogeneous transformation
    
    Eigen::Quaterniond rotation;
    Eigen::Vector3d translation;
    RotationList dQ;
    
    int selected_handle = -1;
    ShaderSettings shader_settings;
    
    void init_scene(const std::vector< std::string > & mesh_paths, const std::string &weight_path);
    void init_scene(const std::string & rest_path, const std::string & result_path);
    void show_pose(const int index_pos);
    void compute_handle_positions();
    void load_scene(std::string path = "");
    void save_scene(std::string path = "");

    void bind_scene_varying_uniforms(GLuint program_id) const;
    
    // Get the screen position given the viewport, the projection*modelview matrix, and the 3D position.
	Eigen::Vector2f screen_position( const Eigen::Matrix4f& mvp, const Eigen::Vector3f& pos ) const
	{
		using namespace Eigen;
		Vector4f clipSpacePos = mvp*pos.homogeneous();
		Vector3f ndcSpacePos = clipSpacePos.head(3) / clipSpacePos(3);
		Vector2f windowSpacePos;
		windowSpacePos(0) = (ndcSpacePos(0) + 1.0) / 2.0 * viewPort(2);
		windowSpacePos(1) = (ndcSpacePos(1) + 1.0) / 2.0 * viewPort(3);
		return windowSpacePos;
	}
    void bind_constant_uniforms( GLuint program_id ) const
	{
		glUseProgram( program_id );
	
		GLint location = uniform(program_id,"num_handles");
		glUniform1i(location, H);
	}
	
	void bind_transforms( GLuint program_id, bool is_animated );
	void build_mesh_program(GLuint &program_id);
	void build_wire_program(GLuint &program_id);
	void load_results(std::string result_path);
	
};

#endif /* __scene_h__ */
