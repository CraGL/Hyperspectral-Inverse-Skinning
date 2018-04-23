from __future__ import print_function, division

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def parseOutput( path ):
	data, costs = [], []
	dim, ortho, handle = None, None, None
	file = open( path, "r" ) 
	for line in file:
		if line.startswith( "ambient dimension: " ):
			newdim = int( line[ len( "ambient dimension: " ): ] )
			assert dim is None or dim == newdim
			dim = newdim
		
		if line.startswith( "given flat orthogonal dimension: " ):
			ortho = int( line[ len( "given flat orthogonal dimension: " ): ] )
		if line.startswith( "affine subspace dimension: " ):
			handle = int( line[ len( "affine subspace dimension: " ): ] )
		
		if line.startswith( "Terminated - " ):
			assert dim is not None
			assert ortho is not None
			assert handle is not None
			
			s = line[ len( "Terminated - " ): ]
			if s.startswith( "max iterations reached" ): 
				## max iterations reached
				data.append( [ dim-ortho, handle, 1000 ] )
			else:
				words = s.split( " " )
				if words[0] == "max":
					## max time reached
					data.append( [ dim-ortho, handle, 1000 ] )
					# data.append( [ dim-ortho, handle, int( words[4] ) ] )
				else:
					## min grad norm reached
					data.append( [ dim-ortho, handle, int( words[5] ) ] )
			
			dim = ortho = handle = None
		
		if line.startswith( "Final cost: " ):
			costs.append( [ float( line[ len( "Final cost: " ): ] ) ] )
	
	data = np.array( data )
	costs = np.array( costs )
	assert len( data ) == len( costs )
	return data, costs
	

if __name__ == '__main__':
	import argparse
	
	parser = argparse.ArgumentParser( description='plot flat intersection difficulies' )
	parser.add_argument( 'path', type=str, help='path of the data file' )
	parser.add_argument( 'which', type=str, default = "error", choices = ['iterations', 'error'], help='Whether to plot "iterations" or "error"' )
	parser.add_argument( '--out', type=str, help='path to save the plot' )
	def str2bool(s): return {'true': True, 'yes': True, 'false': False, 'no': False}[s.lower()]
	parser.add_argument( '--show', type=str2bool, default = True, help='Whether to show the result.' )
	
	print( """Example: python3 plot_flat_difficulites.py test_difficulties/test_flat_difficulites.out-v3 error --show no --out error.pdf
Example: python3 plot_flat_difficulites.py test_difficulties/test_flat_difficulites.out-v3 iterations --show no --out iterations.pdf
""" )
	
	args = parser.parse_args()
	data, costs = parseOutput( args.path )
	
	names = ["given flats dimension $d$", "unknown flat dimension $k$", "number of iterations"]
	df = pd.DataFrame( data, columns=names )	
	df = df.pivot( names[0], names[1], names[2] )
	
	## The order of magnitude is more important:
	# costs = costs.round(3)
	## The numbers between -10 and -30 are distracting in a log plot:
	# costs = np.log10( costs)
	costs = np.log10( costs).clip( -10, None )
	# costs[ costs <= -10 ] = -10
	
	data2 = np.hstack( ( data[:, :2], costs ) )
	df2 = pd.DataFrame( data2, columns=names )
	df2 = df2.pivot( names[0], names[1], names[2] )
	df2.index = df2.index.astype( int )
	df2.columns = df2.columns.astype( int )

	# Draw a heatmap with the numeric values in each cell
	## The default dimensions (9,6) are OK for 12-dimensional data
	## but should be scaled for larger.
	## Scale vertically less; it's only cramped by two-digit labels.
	width = max(9, 9*df.shape[0]/12)
	height = max(6, 6*(df.shape[0]-12)/8)
	print( "width:", width )
	print( "height:", height )
	if args.which == "iterations":
		f, ax = plt.subplots( figsize=(width, height) )
		sns.heatmap(df, annot=True, fmt="d", linewidths=.5, ax=ax)
		ax.invert_yaxis()
	elif args.which == "error":
		f, ax = plt.subplots( figsize=(width, height) )
		sns.heatmap(df2, annot=True, fmt=".2g", linewidths=.5, ax=ax)
		ax.invert_yaxis()

	if args.out:
		print( "Saving", args.out, "..." )
		plt.savefig(args.out)
		print( "Saved:", args.out )
	if args.show: plt.show()
