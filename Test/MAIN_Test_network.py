# Imports
import numpy as np
from numpy import hstack
import matplotlib.pyplot as plt
import random
np.random.seed(1)  

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

from keras.utils.vis_utils import plot_model

from tensorflow import set_random_seed
np.random.seed(1)

# imports from file
from Functions.Data_load    import * 
from Functions.Data_filters import *
from Functions.Data_parser  import *
from Functions.Data_plot    import *

# ========= Settings ========================
random.seed( 30 )
# Disable SettingWithCopyWarning
pd.options.mode.chained_assignment = None  # default='warn'
# Display all columns with pd.head()
pd.set_option('display.max_columns', 50)

# ========= Data Manipulation ===============
# Load in the measurement data
folder = '..\\RawData\\Pendulum_03'
date_train = '2019_ 02_ 27_16_17_33_'
date_test  = '2019_ 02_ 27_16_18_44_'
df_train = Data_mod_Load(folder, date_train)
df_test  = Data_mod_Load(folder, date_test)

# Apply filters
df_train_filtered = Data_mod_Filter(df_train, scale = 1)
df_test_filtered  = Data_mod_Filter(df_test, scale = 1)

#=============================================================================
# Parsing data and hyper parameters
epochs     = 50
n_steps    = 8
features   = 10
output_num = -1 # For example: -1=dz, -8=z

X_train, Y_train = split_sequences(np.array(df_train_filtered), n_steps, features=features, output_num = output_num)
X, Y   = split_sequences(np.array(df_test_filtered),  n_steps, features=features, output_num = output_num)

X_valid = X[0:int(len(X)*0.5),:,:]
Y_valid = Y[0:int(len(Y)*0.5)]
X_test  = X[int(len(X)*0.5):,:,:]
Y_test  = Y[int(len(Y)*0.5):]


#===== Neural Network Modell ================================================
# the dataset knows the number of features, e.g. 2
n_features = X_train.shape[2]

#===== Model Definitions ===================================================
#(uncomment one of the following models)

#===== Model_01 =======================
"""
model = Sequential()

model.add(LSTM(512, activation='relu', return_sequences=False, input_shape=(n_steps, n_features)))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
"""

#===== Model_02 =======================
"""
model = Sequential()
model.add(Conv1D(filters=512, kernel_size=3, activation='relu', input_shape=(n_steps, n_features)))

model.add(LSTM(32, activation='relu', return_sequences=False))

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
"""

##===== Model_03 =======================

model = Sequential()

model.add(Conv1D(filters=512, kernel_size=3, activation='relu', input_shape=(n_steps, n_features)))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
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

#=============================================================================

# fit model
history=model.fit(X_train, Y_train, 
          epochs=epochs,
          validation_data=(X_valid, Y_valid),
          verbose=1,
          shuffle= True)


plt.plot(history.history['loss'],     label='train')
plt.plot(history.history['val_loss'], label='valid')
plt.legend()
plt.show()

model.summary()

# ======== Save model ========================
# save model and architecture to single file (uncomment if needed)
#model.save("network_weights.hdf5")
#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

Y_pred = model.predict(X_test, verbose=1)