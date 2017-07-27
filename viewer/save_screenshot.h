#ifndef __save_screenshot_h__
#define __save_screenshot_h__

#include <string>

// Saves a screenshot to `filename`.
// If `filename` is empty, a unique filename is chosen.
// If `filename` is not empty, saves to that filename.
// If `overwrite` is false and a file with that name exists,
// a number will be appended to make the file name unique.
std::string save_screenshot( const std::string& filename = "", bool overwrite = false );

#endif
