from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

def plot(all_data):
	N = len(all_data)
#	cmap = plt.get_cmap('jet_r')
	cmap = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
	
# 	prop_cycle = plt.rcParams['axes.prop_cycle']
# 	cmap = prop_cycle.by_key()['color']	
	
	legend_y = 0
	tol = 0.0001
	fig = plt.figure() 
#	fig.suptitle(name.title())
	
	num = len(all_data)
	
	for i, item in enumerate(all_data.items()):
		color = cmap[i%N]
		
		legend_name = item[0]
		data = np.array(item[1])
		assert( len(data.shape) == 2 )
		x = range(1,data.shape[1]+1)
		y = np.log(data)
		
		plt.title('Convergence with and without initial guess', fontsize='medium')
		plt.xlabel('iterations')
		plt.ylabel('log(E_RMS)')
		line0, = plt.plot(x, y[0], c=color, lw=2, label=legend_name)
		line1, = plt.plot(x, y[1], c=color, lw=2, label=legend_name+" no initial", ls='dashed')
		plt.plot(x, y[0], 'o', c=color)
		plt.plot(x, y[1], '*', c=color)
			
	plt.legend( loc='upper right', bbox_to_anchor=(1,1), ncol=1, fontsize='x-small')
# 	plt.ylim(ymin=0, ymax=30)
#	plt.xlim(xmin=1)
	plt.savefig('Good convergence with and without initial guess.pdf', bbox_inches='tight')
# 	plt.show()

if __name__ == '__main__':
	import argparse
	
	parser = argparse.ArgumentParser( description='plot convergence speed' )
	parser.add_argument( 'example', type=str, nargs='+', help='folder containinng data files (csv) of one example' )
	
	args = parser.parse_args()
	
	all_data = {}
	import os, sys
	print( "Loading files from: ", args.example )
	for folder in args.example:
		name = folder.split(os.sep)[-1]
		if name == '':	name = folder.split(os.sep)[-2]
		data_file = os.path.join( folder, name+'.csv' )
		data_file_no_init = os.path.join( folder, name+'_no_init.csv' )
		all_data[name] = [np.loadtxt(data_file, delimiter=','), np.loadtxt(data_file_no_init, delimiter=',')]
	
	plot(all_data)