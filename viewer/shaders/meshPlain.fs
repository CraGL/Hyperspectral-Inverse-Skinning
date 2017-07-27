#version 410 core

uniform vec3 eye_pos;
uniform bool enable_color_weight;
uniform int selected_handle;

in vec3 pos_fs_in;
in vec4 color_fs_in;
in vec3 normal_fs_in;
out vec4 frag_color;

void main()
{
	vec3 n = normalize( normal_fs_in );
	vec3 l = normalize( eye_pos - pos_fs_in );	// light direction
	
	// diffuse
	float brightness = max( 0., dot(n, l) );
	
	// specular
	const float shininess = 0.;
	brightness += pow( max( 0., dot( reflect( normalize( eye_pos - pos_fs_in ), n ), l ) ), shininess );
	
	brightness = abs( dot( n, l ) );
	
	frag_color = vec4( 1,1,1,1 );
#ifdef GPU_DEFORM
	if( enable_color_weight ) {
		for(int i=0; i<4; i++) {
			if(selected_handle == indices[i]) {
				float w = weights[i];
				frag_color = vec4(1,0,0,1)*w+vec4(1,1,1,1)*(1-w);
			}
		}
	}
#endif

	frag_color = frag_color * color_fs_in;
	frag_color = frag_color * brightness;
	frag_color.a = 1.0;
	
}
