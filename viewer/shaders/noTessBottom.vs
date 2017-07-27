void main()
{
	pos_fs_in = position;
#ifdef GPU_DEFORM
	pos_fs_in = deform( position );
#else
	pos_fs_in = position;
#endif
	color_fs_in = color;	
	normal_fs_in = normalize( normal );
	gl_Position = modelViewProj * vec4(pos_fs_in, 1.0);	
}