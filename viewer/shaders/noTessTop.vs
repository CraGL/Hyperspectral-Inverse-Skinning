layout(location = POSITION_ATTRIB_LOCATION) in vec3 position;
layout(location = TEXCOORD_ATTRIB_LOCATION) in vec2 vertexUV;
layout(location = NORMAL_ATTRIB_LOCATION) in vec3 normal;
layout(location = COLOR_ATTRIB_LOCATION) in vec4 color;

out vec3 pos_fs_in;
out	vec4 color_fs_in;
out vec3 normal_fs_in;

uniform mat4 modelViewProj;

