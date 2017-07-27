#ifndef __draw_handles__
#define __draw_handles__

#include <Eigen/Core>
#include <igl/Camera.h>

// Draw handles

void init_handles_3d(
	const Eigen::MatrixXd & C,
	const double half_bbd);
	
void end_handles_3d();
	
void draw_handles_3d(
	const igl::Camera & m_camera, 
	const Eigen::MatrixXd & C,
	const Eigen::MatrixXd & T,
	const Eigen::MatrixXf & color);



#endif