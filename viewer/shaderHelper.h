#ifndef __GPU_SHADERHELPER__
#define __GPU_SHADERHELPER__

// #include <OpenGL/gl3.h>
// #include <OpenGL/gl3ext.h>
#include "glcompat.h"
#include <iostream>
#include <vector>

GLint attrib(const GLuint shader, const std::string &name);
GLint uniform(const GLuint shader, const std::string &name);
GLuint createShader_helper(GLint type, const std::string &defines, std::string shader_string, const char* saveAs = nullptr );
GLuint createShader_helper(GLint type, const std::vector< std::string >& shader_strings, const char* saveAs = nullptr );

std::string file_to_string(const std::string &filename);

// Inits the specified type of shader and attaches it to the given program.
// If the program doesn't exist, it is created.
// Returns the shader so you can glDeleteShader() on it after linking. Otherwise,
// you would leak the shader.
// In case of error, returns 0.
GLuint init_shader_type( GLint type, GLuint &program, const std::string &vertex_str );
GLuint init_shader_type( GLint type, GLuint &program, const std::vector< std::string >& vertex_strs );

GLuint init_vertex_shader( GLuint &program, const std::vector< std::string >& vertex_strs );
GLuint init_vertex_shader( GLuint &program, const std::string& vertex_str );

GLuint init_fragment_shader( GLuint &program, const std::string& fragment_str );
GLuint init_fragment_shader( GLuint &program, const std::vector< std::string >& fragment_strs );
GLuint init_geometry_shader( GLuint &program, const std::string& geometry_str );
GLuint init_geometry_shader( GLuint &program, const std::vector< std::string >& geometry_strs );
std::pair< GLuint, GLuint > init_tess_shader( GLuint &program, const std::string &control_str, const std::string &eval_str );
std::pair< GLuint, GLuint > init_tess_shader( GLuint &program, const std::vector< std::string >& control_strs, const std::vector< std::string >& eval_strs );

// init shader from files
// vertex and fragment shader only.
bool build_shaders(
	GLuint &program,
    const std::string &vertex_fname,
    const std::string &fragment_fname);

bool build_shaders_from_strings(
	GLuint &program,
    const std::string &vertex_shader,
    const std::string &fragment_shader);

bool link_shaders( GLuint &program );                    
                    
#endif
