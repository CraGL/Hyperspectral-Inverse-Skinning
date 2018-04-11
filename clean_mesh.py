"""
Compute Convex hull from a set of OBJ poses.

Written by Songrun Liu
"""

from __future__ import print_function, division

import numpy as np
import format_loader
from trimesh import TriMesh

if __name__ == '__main__':
	import os
	import sys
	
	if len(sys.argv) != 3:
		print( "Only need one parameter as input path" )
		exit(-1)
	mesh_path = sys.argv[1]
	output_path = sys.argv[2]
	mesh = TriMesh.FromOBJ_FileName(mesh_path)
	format_loader.write_OBJ( output_path, mesh.vs, mesh.faces )
	
	
	