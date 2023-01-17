from cProfile import label
from lzma import FILTER_LZMA2
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
from sklearn.utils import resample
import os

if __name__ == "__main__":
	# read ex data from file
	m_name	= "random2"
	dir 	= 'EX_OUTPUT'
	file1 	= os.path.join(dir, m_name + "_CGN")
	file2	= os.path.join(dir, m_name + "_MRSM")
	file3	= os.path.join(dir, m_name + "_MDMRM")
	file4 	= os.path.join(dir, m_name + "_BICG") 

	data_paths = [file1, file2, file3, file4]
	# data_paths = [file1, file2, file3]
	# data_paths = [file3, file4]

	# initialize figure and other parameters
	method_names= []
	datas	= []
	N = 0
	d = 0
	m = 0

	# read data from files
	for data_path in data_paths:
		f = open(data_path, 'r')

		data_lines = f.readlines()
		for i in range(0, len(data_lines)):
			data_lines[i] = data_lines[i].rstrip('\n')
		f.close()
		
		# filter comments in data
		filter_data_content = filter(lambda line : line[0] != '#', data_lines)
		data_lines 	= list(filter_data_content)

		# N, d, m 	= data_lines[0].split('\t')
		method_name = data_lines[0]
		method_names.append(method_name)

		file_data		= [[np.double(data_lines[i].split('\t')[0]), 
						np.double(data_lines[i].split('\t')[1])] 
							for i in range(2, len(data_lines))]
		file_data = np.array(file_data)
		datas.append(file_data)

	# draw with SciencePlots
	# with plt.style.context(['science', 'ieee', 'vibrant']):
	with plt.style.context(['science', 'ieee', 'high-vis']):
		fig, ax0 = plt.subplots(nrows = 1, ncols = 1)
		# data setting
		for i, data in enumerate(datas):
			iter_step 	= data[:, 0]
			res 		= data[:, 1]

			"""
			y_interval	= [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, res[1], res[0]]
			y_ticks 	= range(len(y_interval))

			ax0.yaxis.set_ticks(y_ticks)
			ax0.yaxis.set_ticklabels(y_interval)

			y_major_locator = MultipleLocator(5000)
			ax0.yaxis.set_major_locator(y_major_locator)
			"""
			
			ax0.plot(iter_step, res, label = method_names[i])

		# properties setting
		ax0.set_yscale('log')
		# ax0.set_title('{}: {} x {} Matrix with Banded M = {}'.format(m_name, N, N, m))
		# ax0.set_title(m_name)
		ax0.legend(title='Method')
		ax0.set(xlabel='Iteration Step')
		ax0.set(ylabel='Norm 2 of Residual')
		ax0.autoscale(tight=True)
		fig.savefig('EX_OUTPUT/{}.jpg'.format(m_name), dpi=300)

