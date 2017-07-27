layout(location = WEIGHT_INDEX_LOCATION) in vec4 indices;
layout(location = WEIGHT_ATTRIB_LOCATION) in vec4 weights;

const int MAX_HANDLE_NUM = 64;

uniform float threshold;
uniform int num_handles;
uniform mat4 transforms[MAX_HANDLE_NUM];

#define GPU_DEFORM 1

// Get the LBS transform.
// Less efficient if you want the transform times one point.
// If you want to transform multiple points from the same UV, this is probably better.
mat4 lbs_transform( const vec2 uv_coords )
{
	mat4 T = mat4(0.);	
#ifdef NORMALIZE
	float total_w = 0.;
#endif
	for( int i=0; i<4; i++ ) {
		float w = weights[i];
		T += w * transforms[indices[i]];
#ifndef NORMALIZE
    }
    return T;   
#else
		total_w += w;
	}

	if( total_w != 0 )
		return T / total_w;
	else
		return T;
#endif
}

vec3 lbs( vec3 p )
{  
	vec3 deformed_p = vec3(0., 0., 0.);
#ifdef NORMALIZE
	float total_w = 0.;
#endif
	for( int i=0; i<4; i++ ) {
		float w = weights[i];
		T += w * transforms[indices[i]];
#ifdef NORMALIZE
		total_w += w;
#endif
	}

#ifdef NORMALIZE
	if( total_w != 0 ) {
		return deformed_p / total_w;
	} else {
		return p;
	}
#else
    return deformed_p;
#endif
}

// Get the LBS inverse transpose.
mat3 lbs_inverse_transpose_transform( const vec2 uv_coords )
{
	mat3 T = mat3(0.);
	
#ifdef NORMALIZE
	float total_w = 0.;
#endif
	for( int i=0; i<4; i++ ) {
		float w = weights[i];
		T += w * transforms[indices[i]];
#ifdef NORMALIZE
		total_w += w;
#endif
	}

#ifdef NORMALIZE
	if( total_w != 0 ) {
		return inverse(transpose( T*total_w ));
		// return T/total_w;
	} else {
	    // It's a zero matrix; no point inverting or transposing it.
		return T;
	}
#else
    return inverse(transpose(T));
#endif
}

vec3 lbs_normal( const vec3 n, const vec2 uv_coords )
{
    return lbs_inverse_transpose_transform( uv_coords ) * n;
}

// input: unit quaternion 'q0', translation vector 't' 
// output: unit dual quaternion 'dq'
void quatTrans2UDQ(const vec4 q0, const vec3 t, out vec4 dq[2])
{
   // non-dual part (just copy q0):
   dq[0] = q0;
   // dual part:
   dq[1][0] = -0.5*(t[0]*q0[1] + t[1]*q0[2] + t[2]*q0[3]);
   dq[1][1] = 0.5*( t[0]*q0[0] + t[1]*q0[3] - t[2]*q0[2]);
   dq[1][2] = 0.5*(-t[0]*q0[3] + t[1]*q0[0] + t[2]*q0[1]);
   dq[1][3] = 0.5*( t[0]*q0[2] - t[1]*q0[1] + t[2]*q0[0]);
}

// input: dual quat. 'dq' with non-zero non-dual part
// output: unit quaternion 'q0', translation vector 't'
void dQ2QuatTrans(const vec4 dq[2], 
                  out vec4 q0, out vec3 t)
{
   float len = length(dq[0]);
   q0 = dq[0] / len;
   t[0] = 2.0*(-dq[1][0]*dq[0][1] + dq[1][1]*dq[0][0] - dq[1][2]*dq[0][3] + dq[1][3]*dq[0][2]) / len;
   t[1] = 2.0*(-dq[1][0]*dq[0][2] + dq[1][1]*dq[0][3] + dq[1][2]*dq[0][0] - dq[1][3]*dq[0][1]) / len;
   t[2] = 2.0*(-dq[1][0]*dq[0][3] - dq[1][1]*dq[0][2] + dq[1][2]*dq[0][1] + dq[1][3]*dq[0][0]) / len;
}

// input: unit dual quaternion 'dq'
// output: transformation matrix
mat4 dQToMatrix(const vec4 dq[2])
{	
	vec4 q0 = dq[0];
	vec4 q1 = dq[1];
	mat4 M;
	float len2 = dot(q0, q0);
	float w = q0.x, x = q0.y, y = q0.z, z = q0.w;
	float t0 = q1.x, t1 = q1.y, t2 = q1.z, t3 = q1.w;
		
	M[0][0] = w*w + x*x - y*y - z*z; M[1][0] = 2*x*y - 2*w*z; M[2][0] = 2*x*z + 2*w*y;
	M[0][1] = 2*x*y + 2*w*z; M[1][1] = w*w + y*y - x*x - z*z; M[2][1] = 2*y*z - 2*w*x; 
	M[0][2] = 2*x*z - 2*w*y; M[1][2] = 2*y*z + 2*w*x; M[2][2] = w*w + z*z - x*x - y*y;
	
	M[3][0] = -2*t0*x + 2*w*t1 - 2*t2*z + 2*y*t3;
	M[3][1] = -2*t0*y + 2*t1*z - 2*x*t3 + 2*w*t2;
	M[3][2] = -2*t0*z + 2*x*t2 + 2*w*t3 - 2*t1*y;
	
	M /= len2;
	
	M[0][3] = 0.;
	M[1][3] = 0.;
	M[2][3] = 0.;
	M[3][3] = 1.;
	return M;	
}

// input: transformation matrix 'M'
// output: unit dual quaternion 'dq'
void matrixToDq(const mat4 M, out vec4 dq[2])
{
	vec4 q0;
	float w = sqrt(1.0 + M[0][0] + M[1][1] + M[2][2]) / 2.0;
	float w4 = (4.0 * w);
	q0.x = w;
	q0.y = (M[1][2] - M[2][1]) / w4;
	q0.z = (M[2][0] - M[0][2]) / w4;
	q0.w = (M[0][1] - M[1][0]) / w4;
	
	vec3 t = M[3].xyz;
	quatTrans2UDQ(q0, t, dq);
}

// basic dual quaternion skinning:
vec3 dqs( vec3 p)
{
	vec4 blend_dq[2];
	blend_dq[0] = vec4(0.);
	blend_dq[1] = vec4(0.);
	float total_w = 0.;
	for( int i=0; i<4; i++ ) {
		float w = weights[i];
		vec4 dq[2];
		matrixToDq(transforms[i], dq);
		blend_dq[0] += w*dq[0];
		blend_dq[1] += w*dq[1];
		total_w += w;
	}
	blend_dq[0] /= total_w;
	blend_dq[1] /= total_w;
	vec3 deformed_p = (dQToMatrix(blend_dq)*vec4(p,1.0)).xyz;
	return deformed_p;
}

vec3 deform(vec3 p)
{
#ifdef DQS
	return dqs(p);
#else
	return lbs(p);
#endif
}
	