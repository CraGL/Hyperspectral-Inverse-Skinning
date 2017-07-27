#include "glcompat.h"

// Include GLFW
#include <GLFW/glfw3.h>
extern GLFWwindow* window; // The "extern" keyword here is to access the variable "window" declared in tutorialXXX.cpp. This is a hack to keep the tutorials simple. Please avoid this.

#include "common.h"
#include "controls.h"
#include "demo_state.h"
#include "blend_scene.h"

#include <stack>

#include <igl/trackball.h>
#include <igl/two_axis_valuator_fixed_up.h>
#include <igl/snap_to_fixed_up.h>
#include <igl/snap_to_canonical_view_quat.h>
#include <igl/unproject.h>
#include <igl/project.h>

#include <Eigen/Core>
#include <igl/read_triangle_mesh.h>
#include <igl/readTGF.h>

#include <igl/anttweakbar/ReAntTweakBar.h>
#include <igl/REDRUM.h>

///////////////////// global parameters //////////////////////////
float threshold_scale = 2;

std::stack<DemoState> undo_stack;
std::stack<DemoState> redo_stack;

bool command_down = false;
bool shift_down = false;
GLFWcursor* normal_cursor = glfwCreateStandardCursor(GLFW_HRESIZE_CURSOR);  	
GLFWcursor* rotate_cursor = glfwCreateStandardCursor(GLFW_CROSSHAIR_CURSOR);

extern BlendScene g_scene;
extern DemoState g_demo_state;


////////////////////////////////////////////////////////////////////////////

// Type of rotation to use for rotating
enum RotationType
{
  ROTATION_TYPE_IGL_TRACKBALL = 0,
  ROTATION_TYPE_TWO_AXIS_VALUATOR_FIXED_UP = 1,
  NUM_ROTATION_TYPES = 2,
} g_rotation_type = ROTATION_TYPE_TWO_AXIS_VALUATOR_FIXED_UP;

// Type of center to use for rotating
enum CenterType
{
  CENTER_TYPE_ORBIT = 0,
  CENTER_TYPE_FPS  = 1,
  NUM_CENTER_TYPES = 2,
} g_center_type = CENTER_TYPE_ORBIT;

////////////////////////////////////////////////////////////////////////////
int get_window_dpi_scale(GLFWwindow* window)
{
    /*
    Multiply by the result of this function when passing coordinates into AntTweakBar.
    For example, when calling TwEventMousePosGLFW() or TwWindowSize().
    */
    
    // To handle Retina or non-Retina screens, we follow: https://github.com/memononen/nanovg/issues/12
    int win_width, win_height;
	int fb_width, fb_height;
    glfwGetWindowSize(window, &win_width, &win_height);
    glfwGetFramebufferSize(window, &fb_width, &fb_height);
    
    int dpi_scale = (fb_width <= win_width) ? 1 : fb_width / win_width;
    return dpi_scale;
}

// Let x be a world-space point. Then projection*modelview*x is that point in "normalized device coordinates".
// Normalized device coordinates are still not window coordinates, aka positions you get for the mouse in glfw or GLUT.
// This function converts a normalized device coordinate to window coordinates.
void screenFromNDC( const float& xnd, const float& ynd, const GLint* viewport, float& xw, float& yw ) {
	const GLint x = viewport[0];
	const GLint y = viewport[1];
	const GLint width = viewport[2];
	const GLint height = viewport[3];
	
	// man glViewport to find these formulas:
	xw = (xnd+1.)*(width*.5) + x;
	yw = (ynd+1.)*(height*.5) + y;
	
	// screen upper left is (0,0), but window coordinate lower left is (0,0), make it upside down
	yw = height - yw;
}

// Given a `point` and an edge defined by `edge0` and `edge1`, return the distance from
// `point` to the edge.

float point_to_edge_distance( const Eigen::Vector2f& point, 
	const Eigen::Vector2f& edge0, const Eigen::Vector2f& edge1 )
{
	using namespace std;
	using namespace Eigen;
	
	Vector2f p_a = point - edge0;
	Vector2f p_b = point - edge1;
	
	Vector2f b_a = edge1 - edge0;
	Vector2f d = b_a / b_a.norm();
	float cond = p_a.dot( d ) / b_a.norm();
	
	if( cond < 0 )
	{
		return p_a.norm();
	}
	else if( cond > 1 )
	{
		return p_b.norm();
	}
	else
	{
		Vector2f L = p_a - cond * b_a;
		return L.norm();
	}
} 

void resize(GLFWwindow* window, int width, int height)
{
  	g_scene.camera.m_aspect = double(width)/double(height);
  	
  	// Send the new window size to AntTweakBar
  	const int dpi_scale = get_window_dpi_scale(window);
    TwWindowSize(width*dpi_scale, height*dpi_scale);
}

void push_undo() {}
/*
void push_undo(DemoState & _s = g_demo_state)
{
  undo_stack.push(_s);
  // Clear
  redo_stack = std::stack<DemoState>();
}
*/
void undo(GLFWwindow* window)
{
    /*
	if(!undo_stack.empty())
	{
		redo_stack.push(g_demo_state);
		g_demo_state = undo_stack.top();
		undo_stack.pop();
	}
	*/
}
void redo(GLFWwindow* window)
{
    /*
	if(!redo_stack.empty())
	{
		undo_stack.push(g_demo_state);
		g_demo_state = redo_stack.top();
		redo_stack.pop();
	}
	*/
}

// click mouse left key
void left_mouse_down(GLFWwindow* window, int x, int y)
{   
	glfwSetCursor(window, rotate_cursor);
	// collect information for trackball
	g_demo_state.left_mouse_motion = true;
	g_demo_state.m_down_camera = g_scene.camera;
	g_demo_state.m_down_x = x;
	g_demo_state.m_down_y = y;

}

void left_mouse_up(GLFWwindow* window, int x, int y)
{
  	g_demo_state.left_mouse_motion = false;
    glfwSetCursor(window, normal_cursor);
 /*   
	// click not drag
	if( x == g_demo_state.m_down_x && y == g_demo_state.m_down_y )
	{
		using namespace std;
		using namespace Eigen;
		
		const auto& C = g_scene.C;
		const auto& T = g_scene.T;
		const auto& H = g_scene.H;
		
		// Find the positions of handles after transformation.
		MatrixXd TC = MatrixXd::Ones(C.rows(), 4);
		TC.leftCols(3) = C;

		MatrixXd AffinedC = MatrixXd::Ones(C.rows(), 4);
		AffinedC.leftCols(3) = C;
		for( int e=0; e <H; e++ ) {
			Matrix4d Te = Matrix4d::Identity();
			Te.block(0,0,3,4) = T.block(e*4,0,4,3).transpose();
			TC.row(BE(e,0)) = Te*AffinedC.row(BE(e,0)).transpose();
			TC.row(BE(e,1)) = Te*AffinedC.row(BE(e,1)).transpose();
		}

		// Convert bone endpoints from world-space to normalized device coordinates.
		Matrix4d proj = g_scene.camera.projection();
		Matrix4d model = g_scene.camera.inverse().matrix();
		Matrix4d mvp = proj*model;
		
		MatrixXf ndC( C.rows(), 3 );
		for( int i=0; i<C.rows(); i++ ) {
			Vector4f p = (mvp*TC.row(i).transpose()).cast<float>();
			ndC.row(i) = p.head(3)/p(3);
		}

		// Convert from normalized device coordinates to window positions.
		int width, height;
		glfwGetWindowSize(window, &width, &height);
		GLint viewport [4] = {0,0,width,height};
		MatrixXf winC( C.rows(), 2 );
		for( int i = 0; i < C.rows(); ++i ) {
			screenFromNDC( ndC(i,0), ndC(i,1), viewport, winC(i,0), winC(i,1) );
		}
		
		// Now find the closest edge in B to the mouse position in window coordinates.
		float kMouseBonePixelsThreshold = 10;
		int chosen_texture_index = -1;
		float min_dist = width+height;

		for( int bone_index = 0; bone_index < BE.rows(); ++bone_index ) {
			Vector2f p;
			p << x, y;
			float dist = point_to_edge_distance( p, winC.row( BE( bone_index, 0 ) ), winC.row( BE( bone_index, 1 ) ) );
			if( dist < min_dist ) {
				min_dist = dist;
				chosen_texture_index = bone_index;
			}
		}
	
		if( -1 != chosen_texture_index && min_dist < kMouseBonePixelsThreshold ) {
			g_demo_state.sel.resize(1);
			g_demo_state.sel(0) = chosen_texture_index;
			g_scene.selected_handle = chosen_texture_index;
			g_scene.rotation = g_scene.dQ[ chosen_texture_index ];
			cout << "chosen: " << chosen_texture_index << endl;
		} else {
			g_demo_state.sel.resize(1);
			g_demo_state.sel(0) = -1;
			g_scene.selected_handle =  -1;
		}
	}
*/
}

// click mouse right key
void right_mouse_down(GLFWwindow* window, int x, int y)
{
	g_demo_state.right_mouse_motion = true;
	g_demo_state.m_down_x = x;
	g_demo_state.m_down_y = y;
}
void right_mouse_up(GLFWwindow* window, int x, int y)
{
	g_demo_state.right_mouse_motion = false;
}

void rotate(const Eigen::Quaterniond & q_conj)
{
	switch(g_center_type)
	{
		default:
		case CENTER_TYPE_ORBIT:
			g_scene.camera.orbit(q_conj);
			break;
		case CENTER_TYPE_FPS:
			g_scene.camera.turn_eye(q_conj);
			break;
	}
}

void mouse_wheel(GLFWwindow* window, int wheel, int direction, int x, int y)
{
	using namespace Eigen;
	using namespace std;
  
	switch(g_center_type)
	{
		case CENTER_TYPE_ORBIT:
			if(wheel==0)
			{
			// factor of zoom change
			double s = (1.-0.01*direction);
			g_scene.camera.push_away(s);
			}else
			{
			// Dolly zoom:
			g_scene.camera.dolly_zoom((double)direction*1.0);
			}
			break;
		default:
		case CENTER_TYPE_FPS:
			// Move `eye` and `at`
			g_scene.camera.dolly(
			(wheel==0?Vector3d(0,0,1):Vector3d(-1,0,0))*0.1*direction);
			break;
	}
}



void cursor_position_callback(GLFWwindow* window, double x, double y)
{
    using namespace std;
    using namespace Eigen;
	
	const int dpi_scale = get_window_dpi_scale(window);
    if( TwEventMousePosGLFW( x*dpi_scale, y*dpi_scale ) ) {
        // cout << "cursorposition for tw" << endl;
        return;
    }
    
	int width, height;
	glfwGetWindowSize(window, &width, &height);
  
	if(g_demo_state.left_mouse_motion)
	{
		glfwSetCursor(window, rotate_cursor);
		Quaterniond q;
		Quaterniond dq = g_demo_state.m_down_camera.m_rotation_conj;
		switch(g_rotation_type)
		{
			case ROTATION_TYPE_IGL_TRACKBALL:
			{
			// Rotate according to trackball
			igl::trackball(
			  width, height, 
			  2.0, 
			  dq, 
			  g_demo_state.m_down_x, g_demo_state.m_down_y, 
			  x, y, 
			  q);
			  break;
			}
			case ROTATION_TYPE_TWO_AXIS_VALUATOR_FIXED_UP:
			{
			// Rotate according to two axis valuator with fixed up vector
			igl::two_axis_valuator_fixed_up(
			  width,height, 
			  2.0,
			  dq, 
			  g_demo_state.m_down_x, g_demo_state.m_down_y, 
			  x, y, 
			  q);
			break;
			}
			default:
			break;
		}
		rotate(q.conjugate());
	}
	else if(g_demo_state.right_mouse_motion)
	{
		Vector3d translation;
		double xoffset = (g_demo_state.m_down_x-x)/100.;
		double yoffset = (y-g_demo_state.m_down_y)/100.;
		translation << xoffset, yoffset, 0;
		g_demo_state.m_down_x = x;
		g_demo_state.m_down_y = y;
		g_scene.camera.dolly(translation);
	}
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    using namespace std;
	
    if( TwEventMouseButtonGLFW(button,action) ) {
        return;
    }
    
	double xpos, ypos;
	glfwGetCursorPos(window, &xpos, &ypos);
	int pos[2];

    if (button == GLFW_MOUSE_BUTTON_LEFT )
	{
		switch(action)
		{
			case GLFW_PRESS: 
			{
                push_undo();
                left_mouse_down(window, xpos, ypos);
				break;
			}
			case GLFW_RELEASE:
			{
				glfwSetCursor(window, normal_cursor);
				left_mouse_up(window, xpos, ypos);
				break;
			}
        }
    } 
    else if( button == GLFW_MOUSE_BUTTON_RIGHT )
    {
    	switch(action)
		{
			case GLFW_PRESS: 
			{
                push_undo();
                right_mouse_down(window, xpos, ypos);
				break;
			}
			case GLFW_RELEASE:
			{
				glfwSetCursor(window, normal_cursor);
				right_mouse_up(window, xpos, ypos);
				break;
			}
        }
    }
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    double xpos, ypos;
	glfwGetCursorPos(window, &xpos, &ypos);
	
	// Scroll down
	if (yoffset > 0) {
    	mouse_wheel(window, 0,-1,xpos,ypos);
    }
    // Scroll up
    if (yoffset < 0) {
    	mouse_wheel(window, 0,1,xpos,ypos);
    }
    // Scroll left
    if (xoffset < 0) {
    	mouse_wheel(window, 1,-1,xpos,ypos);
    }
    // Scroll right
    if (xoffset > 0) {
    	mouse_wheel(window, 1,1,xpos,ypos);
    }
}

// GLFW error callback
void error_callback(int error, const char* description)
{
    fputs(description, stderr);
}
// GLFW key callback
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if( !TwEventCharGLFW(key, action) )    // Send event to AntTweakBar
    {
        // Event has not been handled by AntTweakBar
        // Do something if needed.
		using namespace std;
	
		if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
			glfwSetWindowShouldClose(window, GL_TRUE);  
		
		if (key == GLFW_KEY_LEFT_SUPER || key == GLFW_KEY_RIGHT_SUPER) {
			if(action == GLFW_PRESS)    	
				command_down = true;
			else if(action == GLFW_RELEASE)
				command_down = false;
		}
			
		if (key == GLFW_KEY_LEFT_SHIFT || key == GLFW_KEY_RIGHT_SHIFT) {
			if(action == GLFW_PRESS)
				shift_down = true;
			else if(action == GLFW_RELEASE)
				shift_down = false;
		}
		
		if(key == GLFW_KEY_Z) {
			if(command_down) {
				if(shift_down) {
					redo(window);
				}
				else {
					undo(window);
				}
			}
			else {
				push_undo();
				Eigen::Quaterniond q;
				igl::snap_to_canonical_view_quat(
				  g_scene.camera.m_rotation_conj,1.0,q);
				rotate(q.conjugate());
			}
		}
	}
}

void character_callback(GLFWwindow* window, unsigned int key)
{

	if( !TwEventCharGLFW(key, 1) )    // Send event to AntTweakBar
    {
        // Event has not been handled by AntTweakBar
        // Do something if needed.
		using namespace std;
		switch(key)
		{
			case '[':
			case ']':
				g_rotation_type = (RotationType)(1-(int)g_rotation_type);
				if(g_rotation_type == ROTATION_TYPE_TWO_AXIS_VALUATOR_FIXED_UP)
				{
					push_undo();
					Eigen::Quaterniond q;
					igl::snap_to_fixed_up(
					  g_scene.camera.m_rotation_conj,
					  q);
					rotate(q.conjugate());
				}
				break;
			case '{':
			case '}':
				g_center_type = (CenterType)(1-(int)g_center_type);
				break;
			case 'r':
			case 'R':
				g_scene.rotation = Eigen::Quaterniond(1,0,0,0);
				break;
			case 'c':
			case 'C':
				g_scene.shader_settings.color_style = static_cast<enum ColorStyleType>((int(g_scene.shader_settings.color_style) + 1)%int(COLOR_TYPE_NUM));
				break;
			case 'z':
			case 'Z':
				break;
			default:
				cout<<"Unknown key command: "<<char(key)<<" "<<key<<endl;
				break;
		}
    }
}
