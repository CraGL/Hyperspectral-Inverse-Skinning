#include "shaderHelper.h"
#include "common.h"

#include <GLFW/glfw3.h>
#include <Eigen/Core>

#include <sstream>
#include <fstream>

namespace
{
    std::string NameFromType( GLint type )
    {
        if (type == GL_VERTEX_SHADER)
            return "Vertex shader";
        else if (type == GL_FRAGMENT_SHADER)
            return "Fragment shader";
        else if (type == GL_GEOMETRY_SHADER)
            return "Geometry shader";
        else if (type == GL_TESS_CONTROL_SHADER)
            return "Tessellation control shader";
        else if (type == GL_TESS_EVALUATION_SHADER)
            return "Tessellation evaluation shader";
        else
            return "Unknown";
    }
}

std::string file_to_string(const std::string &filename) 
{
	std::ifstream t(filename);

	if( !t.good() )
		std::cout << "Fail to read: " << filename << std::endl;
	assert( t.good() );
	
	return std::string((std::istreambuf_iterator<char>(t)),
					   std::istreambuf_iterator<char>());
}

GLint attrib(const GLuint program, const std::string &name)  {
    GLint id = glGetAttribLocation(program, name.c_str());
    if (id == -1)
        std::cerr << ": warning: did not find attrib " << name << std::endl;
    return id;
}

GLint uniform(const GLuint program, const std::string &name)  {
    GLint id = glGetUniformLocation(program, name.c_str());
    if (id == -1) {
        // This is actually desirable behavior,
        // since we want to bind a lot of the same uniforms for multiple shaders,
        // but not all shaders will use them.
        // std::cerr << ": warning: did not find uniform " << name << std::endl;
    }
    return id;
}

GLuint createShader_helper(GLint type, const std::string &defines, std::string shader_string, const char* saveAs) {
    if (shader_string.empty()) {
    	std::cerr << "no shader string." << std::endl;
        return (GLuint) 0;
    }

    if (!defines.empty()) {
        if (shader_string.length() > 8 && shader_string.substr(0, 8) == "#version") {
            std::istringstream iss(shader_string);
            std::ostringstream oss;
            std::string line;
            std::getline(iss, line);
            oss << line << std::endl;
            oss << defines;
            while (std::getline(iss, line))
                oss << line << std::endl;
            shader_string = oss.str();
        } else {
            shader_string = defines + shader_string;
        }
    }
    
    if( saveAs ) { std::ofstream( saveAs ) << shader_string; }

    GLuint id = glCreateShader(type);
    const char *shader_string_const = shader_string.c_str();
    glShaderSource(id, 1, &shader_string_const, nullptr);
    glCompileShader(id);

    GLint status;
    glGetShaderiv(id, GL_COMPILE_STATUS, &status);

    if (status != GL_TRUE) {
        std::cerr << NameFromType( type ) << ":" << std::endl;
        
        std::cerr << "==== SHADER BEGIN" << std::endl;
        std::cerr << shader_string << std::endl << std::endl;
        std::cerr << "==== SHADER END" << std::endl;
        
        char buffer[512];
        glGetShaderInfoLog(id, 512, nullptr, buffer);
        std::cerr << "Error: " << std::endl << buffer << std::endl;
        
        throw std::runtime_error("Shader compilation failed!");
    }

    return id;
}
GLuint createShader_helper(GLint type, const std::vector< std::string >& shader_strings, const char* saveAs ) {
    if (shader_strings.empty()) {
    	std::cerr << "no shader strings." << std::endl;
        return (GLuint) 0;
    }

    GLuint id = glCreateShader(type);
    
    /*
    // I can pass glShaderSource() an array of char*'s like so,
    // but it gives useless line numbers in the error message.
    std::vector< const char* > shader_strings_c;
    shader_strings_c.reserve( shader_strings.size() );
    for( const auto& str : shader_strings ) { shader_strings_c.push_back( str.c_str() ); }
    glShaderSource(id, shader_strings_c.size(), shader_strings_c.data(), nullptr);
    */
    
    // Instead, combine the shaders into one string.
    std::ostringstream shader_strings_together;
    for( const auto& str : shader_strings ) shader_strings_together << str << '\n';
    shader_strings_together << '\n';
    
    const std::string shader_string = shader_strings_together.str();
    const char* shader_string_c = shader_string.c_str();
    
    if( saveAs ) { std::ofstream( saveAs ) << shader_string; }
    
    glShaderSource(id, 1, &shader_string_c, nullptr);
    glCompileShader(id);

    GLint status;
    glGetShaderiv(id, GL_COMPILE_STATUS, &status);

    if (status != GL_TRUE) {
        std::cerr << NameFromType( type ) << ":" << std::endl;
        
        std::cerr << "==== SHADER BEGIN" << std::endl;
        /*
        // I don't need to do this, because I combined them into one string.
        for( const auto& str : shader_strings ) std::cerr << str << '\n';
        std::cerr << std::endl;
        */
        std::cerr << shader_string << std::endl << std::endl;
        std::cerr << "==== SHADER END" << std::endl;
        
        char buffer[512];
        glGetShaderInfoLog(id, 512, nullptr, buffer);
        std::cerr << "Error: " << std::endl << buffer << std::endl;
        
        throw std::runtime_error("Shader compilation failed!");
    }

    return id;
}

GLuint init_shader_type( GLint type, GLuint &program, const std::vector< std::string > &vertex_strs )
{
    GLuint shader = createShader_helper(type, vertex_strs);
    
    if (!shader) {
    	std::cerr << NameFromType(type) << " compiling fails." << std::endl;
        return shader;
    }
    
    if( !glIsProgram( program ) )
	    program = glCreateProgram();
    
    glAttachShader(program, shader);
    
    return shader;
}
GLuint init_shader_type( GLint type, GLuint &program, const std::string &vertex_str )
{
    return init_shader_type( type, program, std::vector< std::string >{ vertex_str } );
}

GLuint init_vertex_shader(  GLuint &program, const std::vector< std::string > &vertex_strs )
{
    return init_shader_type( GL_VERTEX_SHADER, program, vertex_strs );
}
GLuint init_vertex_shader(  GLuint &program, const std::string &vertex_str )
{
    return init_vertex_shader( program, std::vector< std::string >{ vertex_str } );
}

GLuint init_fragment_shader(  GLuint &program, const std::vector< std::string > &fragment_strs )
{
    return init_shader_type( GL_FRAGMENT_SHADER, program, fragment_strs );
}
GLuint init_fragment_shader(  GLuint &program, const std::string &fragment_str )
{
    return init_fragment_shader( program, std::vector< std::string >{ fragment_str } );
}

GLuint init_geometry_shader(  GLuint &program, const std::vector< std::string > &geometry_strs )
{
    return init_shader_type( GL_GEOMETRY_SHADER, program, geometry_strs );
}
GLuint init_geometry_shader(  GLuint &program, const std::string &geometry_str )
{
    return init_geometry_shader( program, std::vector< std::string >{ geometry_str } );
}

// We only make the program if both shaders compile successfully,
// so we can't re-use init_shader_type().
std::pair< GLuint, GLuint >
init_tess_shader( GLuint &program,
			const std::vector< std::string > &control_strs,
			const std::vector< std::string > &eval_strs)
{
    GLuint tessControl =
			createShader_helper(GL_TESS_CONTROL_SHADER, control_strs);
	GLuint tessEval =
			createShader_helper(GL_TESS_EVALUATION_SHADER, eval_strs);
	if (!tessControl || !tessEval) {
		std::cerr << "tessellation shader compiling fails." << std::endl;
		return std::make_pair( tessControl, tessEval );
	}
    
    if( !glIsProgram( program ) )
	    program = glCreateProgram();

	glAttachShader(program, tessControl);
	glAttachShader(program, tessEval);
	
    return std::make_pair( tessControl, tessEval );
}
std::pair< GLuint, GLuint >
init_tess_shader( GLuint &program,
			const std::string &control_str,
			const std::string &eval_str)
{
    return init_tess_shader( program, std::vector< std::string >{ control_str }, std::vector< std::string >{ eval_str } );
}


bool build_shaders(
	GLuint &program,
    const std::string &vertex_fname,
    const std::string &fragment_fname)
{
    return build_shaders_from_strings( program, file_to_string(vertex_fname), file_to_string(fragment_fname) );
}
bool build_shaders_from_strings(
	GLuint &program,
    const std::string &vertex_shader,
    const std::string &fragment_shader)
{
    GLuint mVertexShader =
        createShader_helper(GL_VERTEX_SHADER, "", vertex_shader);
    GLuint mFragmentShader =
        createShader_helper(GL_FRAGMENT_SHADER, "", fragment_shader);
    
    if (!mVertexShader || !mFragmentShader) {
    	if( !mVertexShader ) 
    		std::cerr << "vertex shader compiling fails." << std::endl;
    	if( !mFragmentShader )
    		std::cerr << "fragment shader compiling fails." << std::endl;
        return false;
    }
    program = glCreateProgram();

    glAttachShader(program, mVertexShader);
    glAttachShader(program, mFragmentShader);
    
	const bool result = link_shaders(program);
	
	// Otherwise we would leak these shaders.
	glDetachShader( program, mVertexShader );
	glDetachShader( program, mFragmentShader );
	
	glDeleteShader( mVertexShader );
	glDeleteShader( mFragmentShader );
	
	return result;
}

bool link_shaders( GLuint &program )
{
	glLinkProgram(program);

    GLint status;
    glGetProgramiv(program, GL_LINK_STATUS, &status);

    if (status != GL_TRUE) {
        char buffer[512];
        glGetProgramInfoLog(program, 512, nullptr, buffer);
        std::cerr << "Linker error: " << std::endl << buffer << std::endl;
        program = 0;
        throw std::runtime_error("Shader linking failed!");
    }
    
    glUseProgram(program);
    return true;
}