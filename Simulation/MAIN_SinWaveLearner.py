# Imports
import numpy as np
from numpy import hstack
import matplotlib.pyplot as plt
import random
np.random.seed(1)

from SinFunction import *  

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.callbacks import Callback
from keras.optimizers import SGD
from sklearn import preprocessing
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from sklearn.metrics import mean_squared_error

from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.advanced_activations import LeakyReLU

from keras.utils import plot_model

from tensorflow import set_random_seed
np.random.seed(1)



# Input data generation
dtime  = dTimeGen(dt=0.9, samples=300, noise_offset=0, noise_rate=0.05) # 0.05
atime  = np.cumsum(dtime)
sin_01 = SinGen(atime, off=0, amp=1,   f=0.5,  phase=90, noise_offset=0, noise_rate=0.2) #0.2
sin_02 = SinGen(atime, off=0, amp=1.5, f=0.01,  phase=0,  noise_offset=0, noise_rate=0.01) #0.01

dtime_clc  = dTimeGen(dt=0.9, samples=300, noise_offset=0, noise_rate=0)
atime_clc  = np.cumsum(dtime_clc)
sin_01_clc = SinGen(atime, off=0, amp=1,   f=0.5,  phase=90, noise_offset=0, noise_rate=0)
sin_02_clc = SinGen(atime, off=0, amp=1.5, f=0.01,  phase=0,  noise_offset=0, noise_rate=0)

# Output data generation
dY  = sin_01 + sin_02
cdY = np.cumsum(dY)
iY  = sin_01 * dtime + sin_02
ciY = np.cumsum(iY)

dY_clc  = sin_01_clc + sin_02_clc
cdY_clc = np.cumsum(dY_clc)
iY_clc  = sin_01_clc * dtime + sin_02_clc
ciY_clc = np.cumsum(iY_clc)

# convert to [rows, columns] structure
dtime  = dtime.reshape((len(dtime), 1))
atime  = atime.reshape((len(atime), 1))
sin_01 = sin_01.reshape((len(sin_01), 1))
sin_02 = sin_02.reshape((len(sin_02), 1))

dY = dY.reshape((len(dY), 1))
cdY = cdY.reshape((len(cdY), 1))
iY = iY.reshape((len(iY), 1))
ciY = ciY.reshape((len(ciY), 1))

dtime_clc  = dtime_clc.reshape((len(dtime_clc), 1))
atime_clc  = atime_clc.reshape((len(atime_clc), 1))
sin_01_clc = sin_01_clc.reshape((len(sin_01_clc), 1))
sin_02_clc = sin_02_clc.reshape((len(sin_02_clc), 1))

dY_clc  = dY_clc.reshape((len(dY_clc), 1))
cdY_clc = cdY_clc.reshape((len(cdY_clc), 1))
iY_clc  = iY_clc.reshape((len(iY_clc), 1))
ciY_clc = ciY_clc.reshape((len(ciY_clc), 1))

# horizontally stack columns (change the inputs and output here)
dataset = hstack((dtime, sin_01, sin_02, iY))

epochs     = 50
n_steps    = 8 
n_features = 3

# convert into input/output
X, Y = split_sequences(dataset, n_steps=n_steps, features=n_features, output_num=-1)

# Train - Valid - Test
val_split  = 0.15
test_split = 0.15

atime_cut   = atime_clc[(n_steps-1):]
atime_train = atime_cut[0:int(len(atime_cut)*(1-val_split-test_split))]
atime_valid = atime_cut[int(len(atime_cut)*(1-val_split-test_split)):int(len(atime_cut)*(1-test_split))]
atime_test  = atime_cut[int(len(atime_cut)*(1-test_split)):]
X_train = X[0:int(len(X)*(1-val_split-test_split)),:,:]
X_valid = X[int(len(X)*(1-val_split-test_split)):int(len(X)*(1-test_split)),:,:]
X_test  = X[int(len(X)*(1-test_split)):,:,:]
Y_train = Y[0:int(len(Y)*(1-val_split-test_split))]
Y_valid = Y[int(len(Y)*(1-val_split-test_split)):int(len(Y)*(1-test_split))]
Y_test  = Y[int(len(Y)*(1-test_split)):]
"""
n_seq = 1
X = X.reshape((X.shape[0], n_seq, n_steps, n_features))
"""

# define model
#===== Model_01 =======================
"""
model = Sequential()

model.add(LSTM(512, activation='relu', return_sequences=False, input_shape=(n_steps, n_features)))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
"""

#==== Model_02 ==========================
model = Sequential()
model.add(Conv1D(filters=512, kernel_size=3, activation='relu', input_shape=(n_steps, n_features)))
model.add(Conv1D(filters=512, kernel_size=3, activation='relu'))
model.add(Dropout(0.5))

model.add(TimeDistributed(Flatten()))

model.add(LSTM(512, activation='relu', return_sequences=True))
model.add(LSTM(512, activation='relu', return_sequences=True))
model.add(LSTM(64,  activation='relu', return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32, activation='relu', return_sequences=False))

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')


# fit model
history = model.fit(X_train, Y_train, epochs=epochs,validation_data=(X_test, Y_test), batch_size=16, verbose=1)

model.summary()
#==== Saving model and plotting to file ======================================
# save model and architecture to single file
#model.save("SinWaveLearner_model.h5")

#Plot model to png
#plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
#==============================================================================

# summarize history for accuracy
plt.plot(history.history['loss'],     label='train')
plt.plot(history.history['val_loss'], label='valid')
plt.legend()
plt.show()

# Prediction
Y_pred = model.predict(X_test, verbose=1)
