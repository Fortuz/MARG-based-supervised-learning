"""
array = Data_mod_Parser(df, window=10, features=3)
    df - base data frame
    window - how many time steps will create one sample
    features - how many features are at the begining of the data frame

Info:
    Functions for data parsing
    According to the function parameters provide an array with multiple features
    
    Data Frame and Output Array structures: 
        [features(t)] [outputs(t)]
        [features(t)] [features(t-1)] [features(t-2)] [outputs(t)]
    
"""

import numpy as np

def Data_mod_Parser(df, window=10, features=3):
    df_arr = df.values
    array = np.empty([df_arr.shape[0], df_arr.shape[1]+(((window-1)*features))])
       
    for i in range(window):
        for j in range(features):
            array[:,i*features+j] = np.roll(df_arr[:,j],i,axis=0)
    
    for o in range(df_arr.shape[1]-features):
        array[:,-(o+1)] = df_arr[:,-(o+1)]
    
    #Cut the NaN values
    array = array[(window-1):]
    return array


# split a multivariate sequence into samples
def split_sequences(sequences, n_steps, features, output_num):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :features], sequences[end_ix-1, output_num]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)


"""
For future work:

from pandas import DataFrame
df = DataFrame()
df['t'] = [x for x in range(10)]
df['t-1'] = df['t'].shift(1)
print(df)

It can be done with DataFrames, just add a string properly in every iteration

reference: 
    https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/



from pandas import DataFrame
from pandas import concat
 
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
"""