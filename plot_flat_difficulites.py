import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def parseOutput( path ):
	data, costs = [], []
	ortho, handle = 1, 1
	file = open( path, "r" ) 
	for line in file:
		if line.startswith( "Terminated - " ):
			s = line[ len( "Terminated - " ): ]
			if s.startswith( "max iterations reached" ): 
				## max iterations reached
				data.append( [ ortho, handle, 1000 ] )
			else:
				words = s.split( " " )
				if words[0] == "max":
					## max time reached
					# data.append( [ ortho, handle, 1000 ] )
					data.append( [ ortho, handle, int( words[4] ) ] )
				else:
					## min grad norm reached
					data.append( [ ortho, handle, int( words[5] ) ] )
			handle += 1
			if handle > 12:
				handle = 1
				ortho += 1
			
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
	
	args = parser.parse_args()
	data, costs = parseOutput( args.path )
	
	names = ["orthogonal dimensions", "handles", "number of iterations"]
	df = pd.DataFrame( data, columns=names )	
	df = df.pivot( names[0], names[1], names[2] )
	
	costs = np.log10( costs)
	data2 = np.hstack( ( data[:, :2], costs ) )
	df2 = pd.DataFrame( data2, columns=names )
	df2 = df2.pivot( names[0], names[1], names[2] )
	df2.index = df2.index.astype( int )
	df2.columns = df2.columns.astype( int )

	# Draw a heatmap with the numeric values in each cell
	# f, ax = plt.subplots( figsize=(9, 6) )
	# sns.heatmap(df, annot=True, fmt="d", linewidths=.5, ax=ax)
	f, ax = plt.subplots( figsize=(9, 6) )
	sns.heatmap(df2, annot=True, fmt=".2g", linewidths=.5, ax=ax)

	plt.show()