"""
"""
import numpy as np
import matplotlib.pyplot as plt

# Functions
def dTimeGen(dt, samples=100, noise_offset=0, noise_rate=0.2):
    timing = np.random.normal(noise_offset, noise_rate*dt, samples)
    timing = timing + dt
    return timing

def SinGen(timing, off=0, amp=1, f=1, phase=0, noise_offset=0, noise_rate=0):
    y = []
    for i in range(len(timing)):
        y = (amp * np.sin(f*timing+phase))+off
    noise = np.random.normal(noise_offset, noise_rate*amp, len(timing))
    y = y + noise
    return y

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
dtime = dTimeGen(dt=0.9, samples=50, noise_rate=0)
atime = np.cumsum(dtime)
a = SinGen(atime, off=0, amp=1, f=1, phase=0, noise_rate=0)
b = np.cumsum(a)

plt.plot(atime,  a,  color='green',  label='a')
plt.plot(atime,  b,  color='red',    label='b')
plt.suptitle("Plot", fontsize=16)
plt.legend()
plt.show()
"""