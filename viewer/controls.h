#ifndef __CONTROLS__
#define __CONTROLS__

#include <GLFW/glfw3.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "../json/json.h"

// struct igl_State
// {
//   igl::opengl2::MouseController mouse;
//   Eigen::MatrixXf colors;
// };

// GLFW input controls
// void mouse_wheel(GLFWwindow* window, int wheel, int direction, int x, int y);
// void mouse_drag(GLFWwindow* window, int x, int y);
void resize(GLFWwindow* window, int width, int height);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);
void error_callback(int error, const char* description);
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
void character_callback(GLFWwindow* window, unsigned int key);
int get_window_dpi_scale(GLFWwindow* window);
void rotate(const Eigen::Quaterniond & q_conj);

void windowFromNDC( const float& xnd, const float& ynd, const GLint* viewport, float& xw, float& yw );
float point_to_edge_distance( const Eigen::Vector2f& point, 
	const Eigen::Vector2f& edge0, const Eigen::Vector2f& edge1 );


#endif