import os
import pandas as pd
import matplotlib.pyplot as plt

color_dict_f={'force x': '#62cfac', 'force y': '#6c65b5', 'force z': '#00305e'}

class SimResult:
	def __init__(self, folder):
		self.folder = folder
		self.result = dict(force=pd.DataFrame(), displacement=pd.DataFrame())

	def get_results(self):
		self.result['force'] = pd.read_csv(os.path.join(self.folder, "PCantilevera.sum"),
										   names=['time', 'force x', 'force y', 'force z'], sep=r'\s+')
		self.result['displacement'] = pd.read_csv(os.path.join(self.folder, "PCantilevera.dis"),
												  names=['time', 'displacement x', 'displacement y', 'displacement z'], sep=r'\s+')

	def plot_all_result(self):
		for y in ['force', 'displacement']:
			for col in self.result[y].columns:
				if col != 'time':
					plt.scatter(self.result[y]['time'], self.result[y][col], label=col)
			# Add labels and title
			plt.xlabel('Time')
			plt.ylabel(y)
			plt.title(f'{y} over time')
			plt.legend()
			# Show the plot
			plt.show()
            
	def plot_force_over_time(self, color_dict=color_dict_f, figsize=None, save_fig=None):
		df =  pd.merge(self.result['force'], self.result['displacement'], on="time")
		fig = plt.figure(figsize)
		for key, color in color_dict_f.items():
			plt.plot(df['displacement x'], df[key]*-1, label=key, c=color, marker = '.', linestyle='-')
        
		plt.annotate('displacement z / displecement x = 0.1', xy=(0, -0.3), xycoords='axes fraction', fontsize=12)
		plt.xlabel('displacement [mm]')
		plt.ylabel('reaction force [kN]')
		plt.legend()
		plt.grid()
		plt.legend
		if save_fig:
			plt.savefig(os.path.join('..', 'plots', 'SIM-TRC', save_fig))
		plt.show()