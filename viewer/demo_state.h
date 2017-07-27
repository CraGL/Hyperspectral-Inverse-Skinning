#ifndef __demo_state_h__
#define __demo_state_h__

#include "common.h"
#include <igl/Camera.h>
#include <igl/anttweakbar/ReAntTweakBar.h>

struct DemoState
{
    igl::anttweakbar::ReTwBar rebar;
	igl::Camera m_down_camera;
	bool left_mouse_motion = false;
	bool right_mouse_motion = false;
	int m_down_x, m_down_y;
	Eigen::VectorXi sel;
};

#endif /* __demo_state_h__ */
