from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

def plot(all_data):
	N = len(all_data)
#	cmap = plt.get_cmap('jet_r')
# 	cmap = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
	
	prop_cycle = plt.rcParams['axes.prop_cycle']
	cmap = prop_cycle.by_key()['color']	
	
	legend_y = 0
	fig = plt.figure() 
#	fig.suptitle(name.title())
	
	num = len(all_data)
	
	for i, item in enumerate(all_data.items()):
		color = cmap[i%N]
		
		model_name = item[0]
		data = np.array(item[1])
		assert( len(data.shape) == 2 )
		x = range(data.shape[1])
		y = data
		
		plt.title('Different optimization strategies with performance' , fontsize='medium')
		plt.xlabel('iterations')
		plt.ylabel('E_RMS')
		line1, = plt.plot(x, y[1], c=cmap[1], lw=2, label='Direct(1.5min)')
		line2, = plt.plot(x, y[2], c=cmap[2], lw=2, label='Grassmann(2.2min)')
		line0, = plt.plot(x, y[0], c=cmap[0], lw=2, label='Alternating(1.5min)')
		plt.plot(x, y[0], 'o', c=cmap[0])
		plt.plot(x, y[1], 'o', c=cmap[1])
		plt.plot(x, y[2], 'o', c=cmap[2])
			
	plt.legend( title='cat-poses with 20 handles', loc='upper right', bbox_to_anchor=(1,1), ncol=1, fontsize='medium')
# 	plt.ylim(ymin=0, ymax=30)
#	plt.xlim(xmin=1)
	plt.savefig('strategy_comparison.pdf', bbox_inches='tight')
	plt.show()

if __name__ == '__main__':
	import argparse
	
	parser = argparse.ArgumentParser( description='plot convergence speed with different strategies.' )
	parser.add_argument( 'example', type=str, nargs='+', help='folder containinng data files (csv) of one example' )
	
	args = parser.parse_args()
	
	all_data = {}
	import os, sys
	print( "Loading files from: ", args.example )
	for folder in args.example:
		name = folder.split(os.sep)[-1]
		if name == '':	name = folder.split(os.sep)[-2]
		data_file_biquadratic = os.path.join( folder, name+'_biquadratic.csv' )
		data_file_b = os.path.join( folder, name+'_b.csv' )
		data_file_cayley = os.path.join( folder, name+'_cayley.csv' )
		all_data[name] = [np.loadtxt(data_file_biquadratic, delimiter=','), np.loadtxt(data_file_b, delimiter=','), np.loadtxt(data_file_cayley, delimiter=',')]
	
	plot(all_data)