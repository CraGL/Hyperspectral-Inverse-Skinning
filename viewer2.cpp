// #include <OpenGL/gl3.h>
// #include <OpenGL/gl3ext.h>
#include "viewer/glcompat.h"

// we switch from glut to glfw
#include <GLFW/glfw3.h>

#include <Eigen/Core>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib> // exit()
#include <algorithm> // find()
#include <random>

// Get modification time in a cross-platform way
// From: http://stackoverflow.com/questions/11373505/getting-the-last-modified-date-of-a-file-in-c
#ifdef __APPLE__
#ifndef st_mtime
#define st_mtime st_mtimespec.tv_sec
#endif
#endif

#include "viewer/common.h"
#include "viewer/shaderHelper.h"
#include "viewer/demo_state.h"
#include "viewer/blend_scene.h"
#include "viewer/vertex_array_object.h"
#include "viewer/save_screenshot.h"
#include "viewer/controls.h"
#include "viewer/draw_handles.h"

#include "pythonlike.h"

// #include <igl/read_triangle_mesh.h>
#include <igl/readTGF.h>
#include <igl/pathinfo.h>
#include <igl/bone_parents.h>
#include <igl/colon.h>
#include <igl/forward_kinematics.h>
#include <igl/material_colors.h>
#include <igl/writeOBJ.h>
#include <igl/REDRUM.h>
#include <igl/remove_duplicates.h>
#include <igl/slice.h>
#include <igl/unproject_onto_mesh.h>
#include <igl/project.h>

// Defined in rendering_pipeline.cpp
extern std::string shader_path;

// Globals used by other compilation units.
BlendScene g_scene;
DemoState g_demo_state;

namespace {
    void save_screenshot_cb( void* ) {
        const std::string name = save_screenshot();
        std::cout << "Saved screenshot: " << name << std::endl;
    }
    
    void change_handle_colors( void* ) {
    	std::vector<unsigned int> indices;
    	for(int i=0; i<g_scene.H; i++) {
    		indices.push_back(i);
    	}
    	random_shuffle(indices.begin(), indices.end());
    	g_scene.poses[g_scene.pose_index].color_faces_with_weights(g_scene.W, indices);
    }

	struct RenderState
	{
		bool show_model = true;
		bool show_wireframe = true;
		bool show_handles = true;
		double fps = 0.0;
		double time_stamp = 0;
	};

	RenderState g_render_state;
	bool kTestPerformance = false;
}

// Original Code comes from http://r3dux.org/2012/07/a-simple-glfw-fps-counter/
double calcFPS( double timeInterval, double time_per_frame )
{
    // Static values which only get initialised the first time the function runs
    static double startTime  =  glfwGetTime(); // Set the initial time to now
    // static double fps        =  0.0;           // Set the initial FPS value to 0.0

    // Set the initial frame count to -1.0 (it gets set to 0.0 on the next line). Because
    // we don't have a start time we simply cannot get an accurate FPS value on our very
    // first read if the time interval is zero, so we'll settle for an FPS value of zero instead.
    static double frameCount =  -1.0;
    static double total_fps  = 0.0;
    static int fps_count = 0;

    // Here again? Increment the frame count
    frameCount++;

    // Ensure the time interval between FPS checks is sane (low cap = 0.0 i.e. every frame, high cap = 10.0s)
    if (timeInterval < 0.0)
    {
        timeInterval = 0.0;
    }
    else if (timeInterval > 10.0)
    {
        timeInterval = 10.0;
    }

    // Get the duration in seconds since the last FPS reporting interval elapsed
    // as the current time minus the interval start time
    double duration = glfwGetTime() - startTime;

    // If the time interval has elapsed...
    if (duration > timeInterval)
    {
        time_per_frame = ( duration / frameCount ) * 1000;

        frameCount        = 0.0;
        startTime = glfwGetTime();
    }
    
    return time_per_frame;
}

// GLuint mesh_program(0);
void display(GLFWwindow* window )
{
	using namespace std;
	using namespace Eigen;
	
	/// ================= BEGIN MESH SETUP =================
    GLuint m_VAO(0);        
    auto cleanup_mesh = [&]() {
        // It's not an error to delete 0.
        glBindVertexArray(0);
        glDeleteVertexArrays(1, &m_VAO);
        };
    auto reload_mesh = [&]() {
        cleanup_mesh();
        
//         std::cout << "reload_mesh()" << std::endl;
        // Setup vertex array object.
        m_VAO = general_pipeline_setup_mesh( g_scene.poses[g_scene.pose_index], g_scene.W );
        };
    
    // Do the initial load.
    reload_mesh();
    
	/// ================= END MESH SETUP =================
	
	
	/// ================= BEGIN SHADER SETUP =================
	GLuint mesh_program(0);
    GLuint wire_program(0);
	auto cleanup_shaders = [&]() {
	    // glUseProgram(0) so that we can delete the program right away.
	    glUseProgram(0);
	    
	    // It is not an error to delete 0.
	    glDeleteProgram( mesh_program );
        glDeleteProgram( wire_program );
        
        mesh_program = 0;
        wire_program = 0;
	};
	auto reload_shaders = [&]() {
	    // UPDATE: We don't need to do this. We correctly detach and delete shaders
	    //         from the program. This actually means that the program keeps its
	    //         uniforms (constant ones to need to be set again), but this is called
	    //         infrequently so I won't worry about it.
	    // cleanup_shaders();
	    
	    std::cout << "reload_shaders()" << std::endl;
	    
	    g_scene.build_mesh_program( mesh_program );
	    g_scene.build_wire_program( wire_program );
	    
	    g_scene.bind_constant_uniforms( mesh_program );
	    g_scene.bind_constant_uniforms( wire_program );
	};
	
	// Load the shaders initially.
	// For some reason loading shaders must come after loading the mesh.
	// I'm not sure why, since the only OpenGL calls setting up the Vertex Array Object
	// (which doesn't touch the program) and setting some shader_settings not involved
	// in the #define's.
	reload_shaders();

    /// ================= END SHADER SETUP =================
	
	
	// One-time OpenGL setup
	glfwGetFramebufferSize(window, &g_scene.viewPort(2), &g_scene.viewPort(3));
	glViewport(0, 0, g_scene.viewPort(2), g_scene.viewPort(3));
	
	glClearColor(1.0,1.0,1.0,0.);
	// if( g_scene.shader_settings.tess_style != NO_TESSELLATION )
	glPatchParameteri(GL_PATCH_VERTICES, 3);
	
	// Main display routine
	// Set render_interval to 0 or negative to disable.
    while (!glfwWindowShouldClose(window))
    {   
    	glEnable(GL_DEPTH_TEST);
    	// Accept fragment if it closer to the camera than the former one
		glDepthFunc(GL_LESS); 
		glDisable( GL_POLYGON_OFFSET_FILL );
		
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		
		glEnable( GL_POLYGON_OFFSET_FILL );
		glPolygonOffset( 1, 1 );
		
		reload_mesh();
        if( g_render_state.show_model )
        {	
        	// glDisable(GL_DEPTH_TEST);
			// Use shader and bind changeable uniforms
			glUseProgram( mesh_program );
			g_scene.eye = g_scene.camera.eye().cast<float>();
			g_scene.bind_scene_varying_uniforms(mesh_program);
			g_scene.bind_transforms( mesh_program, false );
			// set drawing parameters
			glEnable(GL_BLEND);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			// glEnable(GL_CULL_FACE);
			// glCullFace(GL_BACK);
	
			// Draw mesh as solid
			glBindVertexArray(m_VAO);
			glDrawElements(GL_TRIANGLES, g_scene.poses[g_scene.pose_index].F.size(), GL_UNSIGNED_INT, 0);

		}
		if( g_render_state.show_wireframe )
		{
			// Bind varying uniforms for wire program
			// bind_wire_varying_uniforms(has_disp);
			glUseProgram( wire_program );
			g_scene.eye = g_scene.camera.eye().cast<float>();
			g_scene.bind_scene_varying_uniforms(wire_program);
			g_scene.bind_transforms(wire_program, false);
		
			// set drawing parameters
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
			
			// Draw mesh as wireframe
			glBindVertexArray(m_VAO);
			glDrawElements(GL_TRIANGLES, g_scene.poses[g_scene.pose_index].F.size(), GL_UNSIGNED_INT, 0);
			
		}
		if( g_render_state.show_handles ) {
			g_scene.compute_handle_positions();
			init_handles_3d(g_scene.C, g_scene.bbd/2);
			
			glDisable(GL_DEPTH_TEST);
			MatrixXf colors = igl::MAYA_VIOLET.transpose().replicate(g_scene.H,1);
			for(int si=0;si<g_demo_state.sel.size();si++)
			{
				int b = g_demo_state.sel(si);
				if( b >= 0 && b < g_scene.H )
				{
					colors.row(b) = igl::MAYA_SEA_GREEN;
				}
			}
			if( g_demo_state.sel.size() > 0 && g_demo_state.sel(0) != -1 ) {
				g_scene.dQ[g_demo_state.sel(0)] = g_scene.rotation;
			}
			draw_handles_3d(g_scene.camera,g_scene.C,g_scene.T,colors);

			glEnable(GL_DEPTH_TEST);
			end_handles_3d();
		}
		
		// Unbind
		glBindVertexArray(0);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glUseProgram(0);
        
        // antTweakBar
        if( !kTestPerformance )
        {		
			TwDraw();
        }
        
        glfwSwapBuffers(window);
        glfwPollEvents();
	}
	
	cleanup_mesh();
    cleanup_shaders();
}

void parse_result(const std::string& result_path, Eigen::MatrixXf& all_transformations) {
}

namespace
{
void usage( const char* argv0 ) {
    std::cerr<<"Usage:"<<std::endl<<"    " << argv0 << " rest_pose.obj result.txt \
    	[--shaders path/to/shaders]"<< std::endl;
    // exit(0) means success. Anything else means failure.
    exit(-1);
}

void manual() {
	std::cout<< R"(
	Usage: ./gpu/demo path/to/mesh.obj path/to/mesh.tgf 
	[Click and drag]        Rotate scene.
	[Vertical scroll]       In orbit mode: zoom in, zoom out.
							In first-person shooter mode: walk in, walk out.
	[Horizontal scroll]     In orbit mode: increase/decrease field of view angle.
							In first-person shooter mode: walk left, walk right.
	[,]                     Toggle between trackball and two-axis-valuator-with-
							fixed-up mouse UI for rotation control.
	{,}                     Toggle between orbiting and first-person shoorter.
	w                       Show wireframe.
	m                       Show model.
	c                       Toggle between no coloring and coloring weights.
	q                       Reset all.
	r                       Reset selected handle transformation.
	s                       Save screenshot of the scene.
	x						Export deformed mesh.
	Z,z                     Snap to canonical view.
	)";
}
}

int main(int argc, char * argv[])
{
	using namespace std;
	using namespace Eigen;
	using namespace pythonlike;
	
	// Always print help information first.
	manual();
	
	/// 1. processing input command.
	vector<string> args( argv + 1, argv + argc );
	
	// Optional arguments.
	get_optional_parameter( args, "--shaders", shader_path );
		
	// Positional arguments.
	if( args.size() != 2 ) {
	    std::cerr << "ERROR: Wrong number of arguments." << std::endl;
	}
	string mesh_path = args.at(0);
	string result_path = args.at(1);
	
	g_scene.init_scene( mesh_path, result_path );
	g_scene.show_pose(0);
	
	// initialize selection
	g_demo_state.sel.resize(1);
	g_demo_state.sel(0) = -1;

	/// 2. initialize GLFW, OPENGL version
	GLFWwindow* window;
    glfwSetErrorCallback(error_callback);
    if (!glfwInit())
        exit(EXIT_FAILURE);
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);
	int g_w=960, g_h=540;
	g_scene.camera.m_aspect = double(g_w)/double(g_h);
    std::string title = os_path_splitext(os_path_split(mesh_path).second).first;
    window = glfwCreateWindow(g_w, g_h, title.c_str(), NULL, NULL);		// "¯\\_(ツ)_/¯"
    if (!window)
    {
		fprintf( stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 4.1 compatible. Try the 2.1 version of the tutorials.\n" );
		glfwTerminate();
        exit(EXIT_FAILURE);
    }
    // Get the actual dimensions of the window we just created.
    glfwGetWindowSize(window, &g_w, &g_h);    
    // set the key callback
    glfwSetKeyCallback(window, key_callback);
   	glfwSetCharCallback(window, character_callback);
	glfwSetWindowSizeCallback(window, resize);
	// set the mouse call back
	glfwSetMouseButtonCallback(window, mouse_button_callback);
	glfwSetScrollCallback(window, scroll_callback);
	glfwSetCursorPosCallback(window, cursor_position_callback);
	
    glfwMakeContextCurrent(window);
	printf("OpenGL version supported by this platform: %s\n", glGetString(GL_VERSION));
	printf("OpenGL renderer: %s\n", glGetString(GL_RENDERER));
	printf("OpenGL vendor: %s\n", glGetString(GL_VENDOR));
	glfwSwapInterval(0);

	/// 3. To handle Retina or non-Retina screens
    // we follow: https://github.com/memononen/nanovg/issues/12
    // NOTE: This fontscaling call to TwDefine must be done before TwInit().
	const int dpi_scale = get_window_dpi_scale(window);
	{
	    int win_width, win_height;
    	int fb_width, fb_height;
	    glfwGetWindowSize(window, &win_width, &win_height);
	    glfwGetFramebufferSize(window, &fb_width, &fb_height);
		
		cout << "Window width x height: " << win_width << " x " << win_height << endl;
		cout << "Framebuffer width x height: " << fb_width << " x " << fb_height << endl;
		cout << "Global width x height: " << g_w << " x " << g_h << endl;
		cout << "dpi_scale: " << dpi_scale << endl;
		assert( g_w == win_width );
		assert( g_h == win_height );
		assert( win_width*dpi_scale == fb_width );
		assert( win_height*dpi_scale == fb_height );
		
		TwWindowSize(fb_width, fb_height);
		
		if(dpi_scale > 1)
        {
            std::stringstream s;
            s << " GLOBAL fontscaling=" << dpi_scale << " ";
            TwDefine(s.str().c_str());
        }
	}
		
    /// 4. Create a tweak bar
 	if( !TwInit(TW_OPENGL_CORE, NULL) && !TwInit(TW_OPENGL, NULL) )
	{
		// A fatal error occured
		fprintf(stderr, "AntTweakBar initialization failed: %s\n", TwGetLastError());
		return 1;
	}
	auto& rebar = g_demo_state.rebar;
	rebar.TwNewBar("properties");
	TwDefine(" properties iconmargin='8 16' ");
	TwDefine(" properties movable=true ");
	TwDefine(" properties resizable=true ");
    TwDefine((" properties size='" + to_string((g_w/4)*dpi_scale) + " " + to_string(g_h*4/5*dpi_scale) + "' ").c_str());
	rebar.TwAddVarRW("camera_rotation", TW_TYPE_QUAT4D,g_scene.camera.m_rotation_conj.coeffs().data(), "readonly=true");
	rebar.TwAddVarRW("eye", TW_TYPE_DIR3F, g_scene.eye.data(), "open readonly=true showval=false");
	rebar.TwAddVarRW("fps", TW_TYPE_DOUBLE, &g_render_state.fps, " label='Milliseconds per frame:' ");
	rebar.TwAddVarRW("show_wireframe",TW_TYPE_BOOLCPP,&g_render_state.show_wireframe,"key=w");
	rebar.TwAddVarRW("show_model",TW_TYPE_BOOLCPP,&g_render_state.show_model,"key=m");
	rebar.TwAddVarRW("show_handles",TW_TYPE_BOOLCPP,&g_render_state.show_handles,"key=h");

	rebar.TwAddVarRW("normalize on-the-fly",TW_TYPE_BOOLCPP,&g_scene.shader_settings.enable_normalize, "");
	
	string pose_selection_text = "rest";
	for(int i=1; i<g_scene.P; i++)		pose_selection_text += (",pos"+to_string(i));
	TwType PoseSelectionTW = igl::anttweakbar::ReTwDefineEnumFromString("PoseSelection",pose_selection_text.c_str()); 
	rebar.TwAddVarRW("Poses",PoseSelectionTW,&g_scene.pose_index,"");
		
	TwType ColoringTypeTW = igl::anttweakbar::ReTwDefineEnumFromString("ColorSchemeType","No Color,Color Weights"); // key = c
	rebar.TwAddVarRW("color scheme",ColoringTypeTW,&g_scene.shader_settings.color_style,"");
	rebar.TwAddVarRW("rotation input", TW_TYPE_QUAT4D, g_scene.rotation.coeffs().data(), "open showval=true");

    rebar.TwAddButton("save screenshot", save_screenshot_cb, nullptr, "key=s");
    rebar.TwAddButton("change colors", change_handle_colors, nullptr, "key=c");
        
	/// 5. display
	display( window );
	
	// clean up
	TwTerminate();
    glfwDestroyWindow(window);
    glfwTerminate();   
    
    exit(EXIT_SUCCESS);

}
