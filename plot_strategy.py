from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

def plot(all_data):
	N = len(all_data)
#	cmap = plt.get_cmap('jet_r')
	cmap = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange']
	
	# prop_cycle = plt.rcParams['axes.prop_cycle']
	# cmap = prop_cycle.by_key()['color']	
	
	legend_y = 0
	fig = plt.figure() 
#	fig.suptitle(name.title())
	
	num = len(all_data)
	
	for i, item in enumerate(all_data.items()):	
		words = item[0].split('-')
		handles = words[-1]
		name = words[0]
		if len(words)>2:	name += '-'+words[1]
		
		y = item[1]
		

		plt.title('Different optimization strategies with performance' , fontsize='medium')
		plt.xlabel('iterations')
		plt.ylabel('Vertex Error')


# ############ without initial value

# 		### cat
# 		plt.plot(range(len(y[0])), y[0], c=cmap[0], lw=2, label='Biquadratic (2.07 min)')
# 		plt.plot(range(len(y[1])), y[1], c=cmap[1], lw=2, label='pB (1.67 min)')
# 		plt.plot(range(len(y[2])), y[2], c=cmap[2], lw=2, label='IPCA (2.07 min)')
# 		plt.plot(range(len(y[3])), y[3], c=cmap[3], lw=2, label='pymanopt_pB_conjugate (7.28 min)')
# 		plt.plot(range(len(y[4])), y[4], c=cmap[4], lw=2, label='pymanopt_pB_trust (23.28 min)')
# 		plt.plot(range(len(y[5])), y[5], c=cmap[5], lw=2, label='pymanopt_pB_steepest (6.71 min)')
# 		plt.plot(range(len(y[6])), y[6], c=cmap[6], lw=2, label='intersection_conjugate (16.52 min)')
# 		plt.ylim(ymin=0, ymax=220)

		
# 		# ### cheburashka
# 		# plt.plot(range(len(y[0])), y[0], c=cmap[0], lw=2, label='Biquadratic (1.49 min)')
# 		# plt.plot(range(len(y[1])), y[1], c=cmap[1], lw=2, label='pB (1.26 min)')
# 		# plt.plot(range(len(y[2])), y[2], c=cmap[2], lw=2, label='IPCA (1.94 min)')
# 		# plt.plot(range(len(y[3])), y[3], c=cmap[3], lw=2, label='pymanopt_pB_conjugate (5.99 min)')
# 		# plt.plot(range(len(y[4])), y[4], c=cmap[4], lw=2, label='pymanopt_pB_trust (19.94 min)')
# 		# plt.plot(range(len(y[5])), y[5], c=cmap[5], lw=2, label='pymanopt_pB_steepest (5.73 min)')
# 		# plt.plot(range(len(y[6])), y[6], c=cmap[6], lw=2, label='intersection_conjugate (17.62 min)')
# 		# plt.ylim(ymin=0, ymax=140)


# 		plt.plot(range(len(y[0])), y[0], 'o', c=cmap[0])
# 		plt.plot(range(len(y[1])), y[1], 'o', c=cmap[1])
# 		plt.plot(range(len(y[2])), y[2], 'o', c=cmap[2])
# 		plt.plot(range(len(y[3])), y[3], 'o', c=cmap[3])
# 		plt.plot(range(len(y[4])), y[4], 'o', c=cmap[4])
# 		plt.plot(range(len(y[5])), y[5], 'o', c=cmap[5])
# 		plt.plot(range(len(y[6])), y[6], 'o', c=cmap[6])



############ with initial guess. So laplacian choice is included.
# ## cylinder
# 		plt.plot(range(len(y[0])), y[0], c=cmap[0], lw=2, label='Biquadratic (0.08 min)')
# 		plt.plot(range(len(y[1])), y[1], c=cmap[1], lw=2, label='pB (0.06 min)')
# 		plt.plot(range(len(y[2])), y[2], c=cmap[2], lw=2, label='IPCA (0.07 min)')
# 		plt.plot(range(len(y[3])), y[3], c=cmap[3], lw=2, label='pymanopt_pB_conjugate (0.29 min)')
# 		plt.plot(range(len(y[4])), y[4], c=cmap[4], lw=2, label='pymanopt_pB_trust (17.41 min)')
# 		plt.plot(range(len(y[5])), y[5], c=cmap[5], lw=2, label='pymanopt_pB_steepest (0.33 min)')
# 		plt.plot(range(len(y[6])), y[6], c=cmap[6], lw=2, label='intersection_conjugate (0.33 min)')
# 		plt.plot(range(len(y[7])), y[7], c=cmap[7], lw=2, label='laplacian (0.34 min)')
# 		plt.ylim(ymin=0, ymax=370)


## cat
		plt.plot(range(len(y[0])), y[0], c=cmap[0], lw=2, label='Biquadratic (1.91 min)')
		plt.plot(range(len(y[1])), y[1], c=cmap[1], lw=2, label='pB (1.55 min)')
		plt.plot(range(len(y[2])), y[2], c=cmap[2], lw=2, label='IPCA (1.90 min)')
		plt.plot(range(len(y[3])), y[3], c=cmap[3], lw=2, label='pymanopt_pB_conjugate (6.84 min)')
		plt.plot(range(len(y[4])), y[4], c=cmap[4], lw=2, label='pymanopt_pB_trust (18.87 min)')
		plt.plot(range(len(y[5])), y[5], c=cmap[5], lw=2, label='pymanopt_pB_steepest (6.64 min)')
		plt.plot(range(len(y[6])), y[6], c=cmap[6], lw=2, label='intersection_conjugate (17.61 min)')
		# plt.plot(range(len(y[7])), y[7], c=cmap[7], lw=2, label='laplacian (?? min)')
		plt.ylim(ymin=0, ymax=45)


# ### cheburashka
# 		plt.plot(range(len(y[0])), y[0], c=cmap[0], lw=2, label='Biquadratic (1.57 min)')
# 		plt.plot(range(len(y[1])), y[1], c=cmap[1], lw=2, label='pB (1.20 min)')
# 		plt.plot(range(len(y[2])), y[2], c=cmap[2], lw=2, label='IPCA (2.15 min)')
# 		plt.plot(range(len(y[3])), y[3], c=cmap[3], lw=2, label='pymanopt_pB_conjugate (6.00 min)')
# 		plt.plot(range(len(y[4])), y[4], c=cmap[4], lw=2, label='pymanopt_pB_trust (18.58 min)')
# 		plt.plot(range(len(y[5])), y[5], c=cmap[5], lw=2, label='pymanopt_pB_steepest (6.29 min)')
# 		plt.plot(range(len(y[6])), y[6], c=cmap[6], lw=2, label='intersection_conjugate (17.77 min)')
# 		# plt.plot(range(len(y[7])), y[7], c=cmap[7], lw=2, label='laplacian (?? min)')
# 		plt.ylim(ymin=0, ymax=30)
		

		plt.plot(range(len(y[0])), y[0], 'o', c=cmap[0])
		plt.plot(range(len(y[1])), y[1], 'o', c=cmap[1])
		plt.plot(range(len(y[2])), y[2], 'o', c=cmap[2])
		plt.plot(range(len(y[3])), y[3], 'o', c=cmap[3])
		plt.plot(range(len(y[4])), y[4], 'o', c=cmap[4])
		plt.plot(range(len(y[5])), y[5], 'o', c=cmap[5])
		plt.plot(range(len(y[6])), y[6], 'o', c=cmap[6])
		# plt.plot(range(len(y[7])), y[7], 'o', c=cmap[7])


		
		plt.legend( title=name+' with '+handles+' handles'+ ' and 0 as seed', loc='upper right', bbox_to_anchor=(1,1), ncol=1, fontsize='medium')
		plt.savefig('strategy_comparison_'+ item[0] +'.pdf', bbox_inches='tight')
		plt.show()

if __name__ == '__main__':
	import argparse
	
	parser = argparse.ArgumentParser( description='plot convergence speed with different strategies.' )
	parser.add_argument( 'example', type=str, nargs='+', help='folder containinng data files (csv) of one example' )
	
	args = parser.parse_args()
	
	import os, sys
	all_data = {}
	print( "Loading files from: ", args.example )
	for folder in args.example:
		name = folder.split(os.sep)[-1]
		if name == '':	name = folder.split(os.sep)[-2]
		data_file_biquadratic = os.path.join( folder, name+'_biquadratic_0.csv' )
		data_file_b = os.path.join( folder, name+'_b_0.csv' )
		data_file_ipca = os.path.join( folder, name+'_ipca_0.csv' )
		data_file_pB_pymanopt_conjugate = os.path.join( folder, name+'_pB_pymanopt_conjugate_0.csv' )
		data_file_pB_pymanopt_trust = os.path.join( folder, name+'_pB_pymanopt_trust_0.csv' )
		data_file_pB_pymanopt_steepest = os.path.join( folder, name+'_pB_pymanopt_steepest_0.csv' )
		data_file_intersection_conjugate = os.path.join( folder, name+'_intersection_conjugate_0.csv' )
		data_file_laplacian = os.path.join( folder, name+'_laplacian_0.csv' )


		all_data[name] = [np.loadtxt(data_file_biquadratic, delimiter=',')[:20], np.loadtxt(data_file_b, delimiter=',')[:20], np.loadtxt(data_file_ipca, delimiter=',')[:20], np.loadtxt(data_file_pB_pymanopt_conjugate, delimiter=',')[:20], np.loadtxt(data_file_pB_pymanopt_trust, delimiter=',')[:20], np.loadtxt(data_file_pB_pymanopt_steepest, delimiter=',')[:20], np.loadtxt(data_file_intersection_conjugate, delimiter=',')[:20]]

		# all_data[name] = [np.loadtxt(data_file_biquadratic, delimiter=',')[:20], np.loadtxt(data_file_b, delimiter=',')[:20], np.loadtxt(data_file_ipca, delimiter=',')[:20], np.loadtxt(data_file_pB_pymanopt_conjugate, delimiter=',')[:20], np.loadtxt(data_file_pB_pymanopt_trust, delimiter=',')[:20], np.loadtxt(data_file_pB_pymanopt_steepest, delimiter=',')[:20], np.loadtxt(data_file_intersection_conjugate, delimiter=',')[:20], np.loadtxt(data_file_laplacian, delimiter=',')[:20] ]
	
	plot(all_data)










