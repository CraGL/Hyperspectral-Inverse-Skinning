#include "save_screenshot.h"

// #include <OpenGL/gl3.h>
#include "glcompat.h"

// We now have a "stb_image_write.cpp" in utils so that multiple files can use stb_image_write.h.
// #define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "pythonlike.h"

#include <cassert>
#include <vector>

// now_as_string() needs functions from here.
#include <ctime>

namespace {
std::string now_as_string()
{
    // From: http://stackoverflow.com/questions/5438482/getting-the-current-time-as-a-yyyy-mm-dd-hh-mm-ss-string
    std::time_t rawtime;
    std::tm* timeinfo;
    char buffer[80];

    std::time( &rawtime );
    timeinfo = std::localtime( &rawtime );
    
    std::strftime( buffer, 80, "%Y-%m-%d-%H-%M-%S", timeinfo );
    
    return std::string( buffer );
}

std::string unique_name( const std::string& desired_name )
{
    using namespace pythonlike;
    
    std::string name = desired_name;
    int num = 0;
    while( os_path_exists( name ) ) {
        num += 1;
        auto split = os_path_splitext( name );
        name = split.first + " " + std::to_string( num ) + split.second;
    }
    
    return name;
}

// Flip the image in its y-axis.
void flip_inplace( void* image, int width, int height, int pixelbytes )
{
    // Someone else can generalize this.
    assert( 4 == pixelbytes );
    
    typedef uint32_t Pixel;
    
    Pixel temp;
    
    for( int row = 0; row < height/2 ; ++row )
	for( int col = 0; col < width; ++col )
	{
		Pixel* lhs = static_cast< Pixel* >( image ) + col + width*row;
		Pixel* rhs = static_cast< Pixel* >( image ) + col + width*( height - row - 1 );
		
		Pixel temp = *lhs;
		
		*lhs = *rhs;
		*rhs = temp;
	}
}
}

// Saves a screenshot to `filename`.
// If `filename` is empty, a unique filename is chosen.
// If `filename` is not empty, saves to that filename.
// If `overwrite` is false and a file with that name exists,
// a number will be appended to make the file name unique.
// Returns the saved filename.
std::string save_screenshot( const std::string& filename, bool overwrite )
{
    std::string name = filename;
    if( name.empty() ) {
        name = "screenshot " + now_as_string() + ".png";
        assert( !overwrite );
    }
    
    if( !overwrite ) {
        name = unique_name( name );
    }
    
    {
        GLint view[4];
        glGetIntegerv( GL_VIEWPORT, view );
        
        std::vector< unsigned char > img( view[2]*view[3]*4 );
        glReadBuffer( GL_FRONT );
        glReadPixels( view[0], view[1], view[2], view[3], GL_RGBA, GL_UNSIGNED_BYTE, img.data() );
        glReadBuffer( GL_BACK );
        
        // OpenGL saves flipped images.
        flip_inplace( img.data(), view[2], view[3], 4 );
        
        stbi_write_png( name.c_str(), view[2], view[3], 4, img.data(), 0 );
    }
    
    return name;
}
