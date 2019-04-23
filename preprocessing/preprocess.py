# Importing the dependencies
import pandas as pd
import numpy as np

# TODO: Make _ for all sequences 
# 1 dimension sequence = multivariate series sample for an activity, flattened into (t*m) vector
# 2 dimension sequence = multivariate series grid for an activity, seen as (t*m) 2D grid with 't' time steps and 'm' features   

# Preprocessor Class
class SequenceHandler:
	def __init__(self, data_path, size=26, debug=False):
		print('Extracting sequences ...', flush=True)
		self._df = pd.read_csv(data_path)
		self._sequence_id_appender()
		self._sequence_generator()
		print(f'Adjusting sequence length to {size} ...', flush=True)
		self._sequence_length_adjuster(size, debug)
		print('Sequence data acquired. (Now convert to appropiate dimension)', flush=True)

	def _sequence_id_appender(self):
		count, volunteer_id = 1, 1 
		seq_indicator = [count] 

		# Record changes in the activity sequence as recorded for an volunteer 
		for row in range(1,len(self._df)):
		    if self._df.loc[row, 'Activity'] == self._df.loc[row-1, 'Activity'] and \
		    volunteer_id == self._df.loc[row, 'subject'] :
		        seq_indicator.append(count) 
		    else : 
		        count, volunteer_id = count + 1, self._df.loc[row, 'subject'] # update the indicators 
		        seq_indicator.append(count)

		self._df.drop(columns=['sequenceInd'], inplace=True) if 'sequenceInd' in self._df else None # remove column if already exists while testing        
		self._df.insert(loc=self._df.shape[1]-2, column='sequenceInd', value=seq_indicator)

	def _sequence_generator(self):
		self.sequence = {}
		groupObj = self._df.groupby(['sequenceInd','subject'])
		for sequence_id, x in enumerate(groupObj.groups):
		    self.sequence[f'{sequence_id+1}'] = groupObj.get_group(x)
		#self._df.loc[343:350,['sequenceInd','Activity']]

	def _add_nan_rows(self, sequence_id, cutoff):#, debug=0):
		num_row, num_col = self.sequence[sequence_id].shape[0], self.sequence[sequence_id].shape[1]
		empty_row = pd.Series([np.NaN]*num_col, index=self.sequence[sequence_id].columns.values.tolist())
		#print(self.sequence[sequence_id]) if debug else None
		self.sequence[sequence_id] = self.sequence[sequence_id].append([empty_row]*(cutoff - num_row), ignore_index=True)

	def _remove_extra_rows(self, sequence_id, cutoff):
		num_row = self.sequence[sequence_id].shape[0]
		self.sequence[sequence_id] = self.sequence[sequence_id][:-(num_row - cutoff)]

	def _fill_missing_values(self, sequence_id):
		self.sequence[sequence_id].iloc[:,:-3] = self.sequence[sequence_id].iloc[:,:-3].interpolate(method='spline', order=1, axis=0)

	def _sequence_length_adjuster(self, cutoff, debug):
		for seq_id, seq_df in self.sequence.items():
			if seq_df.shape[0] < cutoff:
				print(f'Seq_ID - {seq_id} : < cutoff : Before Size - {seq_df.shape[0]} : ', end='') if debug else None
				self._add_nan_rows(seq_id, cutoff)
				self._fill_missing_values(seq_id)
				print(f'After Size - {self.sequence[seq_id].shape[0]}') if debug else None
			elif seq_df.shape[0] > cutoff:
				print(f'Seq_ID - {seq_id} : > cutoff : Before Size - {seq_df.shape[0]} : ', end='') if debug else None
				self._remove_extra_rows(seq_id, cutoff)
				print(f'After Size - {self.sequence[seq_id].shape[0]}') if debug else None
			else:
				print(f'Seq_ID - {seq_id} : == cutoff : Same Size - {seq_df.shape[0]} ') if debug else None

	def convert_sequence_to_1D_sample(self):
		print('Generating 1D view of sequence data ...', flush=True)
		seq_num_rows, seq_num_cols = 0,0
		old_colnames = [] 		
		# Loop just to get sequence dataframe dimensions and column names
		for seq_id, seq_df in self.sequence.items():
			seq_num_rows, seq_num_cols = seq_df.shape
			old_colnames = seq_df.columns.values.tolist()
			break

		# Defining sequence-1d shape before for efficiency 
		self.sequence_1d = np.empty((len(self.sequence), (seq_num_cols-3)*seq_num_rows+3), dtype='O') # dimensions like this because we need only 1 vector copy of the last 3 columns (and not a matrix copy) 
		iter_index = 0

		# Creates the sequence-1d numpy array 
		for seq_id, seq_df in self.sequence.items():
			# Flatten series as 1D features
			features = seq_df.iloc[:,:-3].values.flatten('F')
			features = features.reshape((1, features.shape[0]))

			# Extract seq_id + subject_id + classes
			extras = seq_df.iloc[0,-3:].values
			extras = extras.reshape((1, extras.shape[0]))

			# Create a row and store it in the numpy 1d sequence array
			row = np.concatenate((features, extras),axis=1)
			self.sequence_1d[iter_index,:] = row
			iter_index = iter_index + 1

		# Creates the sequence-1d dataframe by appending column names
		new_colnames = ['']*(seq_num_rows*(seq_num_cols-3)+3) # initializing
		index = 0

		# Loop everything but last three non-feature column names
		for i in range(0, len(old_colnames)-3):
		    for j in range(0,seq_num_rows):
		        new_colnames[index] = f'lag({old_colnames[i]}, {j})' if j != 0 else old_colnames[i]
		        index = index + 1

		# Store last three non-feature column names
		new_colnames[-3:] = ('sequenceInd', 'subject', 'Activity')

		# Save as dataframe	
		self.sequence_1d = pd.DataFrame(data=self.sequence_1d, columns=new_colnames)
		print(f'Sequence samples generated - {self.sequence_1d.shape[0]} sequences rows,  {self.sequence_1d.shape[1]-3} feature columns + 2 helper columns + 1 class column. \n(Now sequence 1-dim is accessible) ', flush=True) 

	def convert_sequence_to_2D_sample(self):
		print('Generating 2D view of sequence data ...', flush=True)
		seq_num_rows, seq_num_cols = 0,0

		# Fetch sequence df dimensions
		for seq_id, seq_df in self.sequence.items():
			seq_num_rows, seq_num_cols = seq_df.shape
			break

		# Intializing stuff
		num_samples = len(self.sequence)
		self.sequence_2d_x, self.sequence_2d_y = np.zeros((num_samples, seq_num_rows, seq_num_cols-3)), ['']*num_samples

		index_sample = 0
		for seq_id, seq_df in self.sequence.items():
			self.sequence_2d_x[index_sample, :,:] = seq_df.iloc[:,:-3].values
			self.sequence_2d_y[index_sample] = seq_df.iloc[0,-1]
			index_sample = index_sample + 1
		print(f'Sequence samples generated - design matrix (#samples, #time_steps, #features) = {self.sequence_2d_x.shape}, class vector (#samples) = ({len(self.sequence_2d_y)}). (Now sequence 2-dim is accessible) ', flush=True)

	def get_sequence(self, dims=1):
		if dims == 1:
			return self.sequence_1d # sequences in 1-dimension 
		if dims ==2:
			return (self.sequence_2d_x, self.sequence_2d_y) # sequence in 2-dims
		else:
			return self.sequence # dictionary of all sequences in 2-dimensions


	#def _standardize_all_data(self):



if __name__ == '__main__': main()