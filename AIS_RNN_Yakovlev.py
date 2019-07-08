from math import sqrt
from numpy import concatenate
import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas import concat
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import AIS_Path_Utils as ais_utils

from keras import Sequential
from keras.layers import Dense, LSTM

ais_data = pd.read_csv("trimmed_M5_Z15.csv", header=0, index_col=0)

### Collection of pandas dataframes for ship paths
paths = {}

mmsi_groups = ais_data.groupby(ais_data.index) #"MMSI") # Split into all the MMSIs, remove MMSI column, sort by time, and reset index
for MMSI in mmsi_groups.groups:
	paths[MMSI] = ais_data[ais_data.index == MMSI].sort_values(by="BaseDateTime").reset_index(drop=True) #.drop(["MMSI"], axis=1).sort_values(by="BaseDateTime").reset_index(drop=True)

''' uncomment this to block plotting
path_iterator = iter(paths)
start = 300
for i in range(start):
	current_ship_course = next(path_iterator) # skip to that index
for i in range(start, start+100):
	current_ship_course = next(path_iterator)
	values = paths[current_ship_course].values
	plt.subplot(10, 10, i-start+1)
	plt.axis('off')
	plt.plot(values[:, 1], values[:, 2])
	if (np.isnan(values[0, 6]) == True):
		plt.title(ais_utils.resolve_vessel_type(values[0, 6])) 
	else:
		plt.title(str(int(values[0, 6])) + ": " + ais_utils.resolve_vessel_type(values[0, 6]))
plt.show() # Show off the first 25 ship tracks

'''
'''
first_mmsi = next(iter(paths)) # get the first key in MSSIs for example plot
i = 1
values = paths[first_mmsi].values
plt.plot(values[:, 1], values[:, 2])
plt.show()


i += 1 #index for subplotting
groups = [1, 2, 3, 4, 5]
for group in groups:
	plt.subplot(len(groups), 1, i)
	plt.plot(values[:, group])
	plt.title(paths[first_mmsi].columns[group], y=0.5, loc="right")
	i+=1
plt.show()
'''

def series_to_supervised(data, n_in=1, n_out=1, dropNaN=True):
	"""
	Frame a time series as a supervised learning dataset
	Arguments:
		data: sequence of observations as a list
		n_in: number of lag observations
		n_out: number of observations as output
		dropNaN: whether to drop rows with NaN values
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()

	# input sequence (t-n ... ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

	#forecast sequence (t, t+1, ... ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [("var%d(t)" % (j+1)) for j in range(n_vars)]
		else:
			names += [("var%d(t+%d)" % (j+1, i)) for j in range(n_vars)]

	# put everything together
	agg = concat(cols, axis=1)
	agg.columns = names

	# drop rows with NaN values
	if dropNaN:
		agg.dropna(inplace=True)
	return agg

# load single dataset
dataset = paths[next(iter(paths))]
values = dataset.values
# integer encode direction
encoder = LabelEncoder()
values[:,4] = encoder.fit_transform(values[:,4])
# ensure all data is float
values = values.astype("float32")
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
# drop columns we don't want to predict
reframed.drop(reframed.columns[9:], axis=1, inplace=True) #,10,11,12,13,14,15]], axis=1, inplace=True)
print(reframed.head)


### Here is the data shaping
# 	Data here is modified to be input into the RNN
values = reframed.values
n_train_hours = 365*24
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]

#split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]


### Reshape input to be 3D (Samples, timesteps, features)
#	Results in (8760, 1, 8) for the first path
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1])) #, train_X.shape[1] - I forgot these are technically 3D arrays
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

### Defining the RNN's structure
#	This is where the magic happens, that sweet keras goodness
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
#model.add(LSTM(50, return_sequences=True))
#model.add(LSTM(50))
model.add(Dense(units=1)) # add one fully connected layer at the end

model.compile(optimizer="adam", loss="mean_squared_error") # Actually put the thing together on Keras' end

model.summary() # Just for our records

### fit network
print("-------------------------- Fitting model ----------------------------")
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
### plot history of training
#plt.plot(history.history['loss'], label='train')
#plt.plot(history.history['val_loss'], label='test')
#plt.legend()
#plt.show()
	
### Make a prediction
print("-------------------------- Making Prediction --------------------------")
yhat = model.predict([test_X[0], test_X[1]])

y0_hat = model.predict(test_X[0])
y1_hat = model.predict(test_X[1])

test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
scaler = scaler.fit(inv_yhat)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]

# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]

arr = []
for i in range(len(test_X[1])):
	arr.append([])
for i in range(len(test_X)-1):
	for j in range(len(arr)-1):
		print(str(i) + ", " + str(j))
		arr[j].append(test_X[i][j])
	
#for i in range(len(test_X[1])):
#	plt.plot(arr[i], label=i)
plt.plot(arr[1], label='mystery')

#plt.plot(test_X[1], label='test_X')
plt.plot(yhat, color='red', label='prediction')
plt.legend()
plt.show()

# calculate Root Mean Squared Error
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

### NOT YET READY
#model = Sequential()
#model.add(LSTM(units=50, return_sequences=True, input_shape=X.shape[1])) # this needs to be modified
#model.add(LSTM(units=50, return_sequences=True))
#model.add(LSTM(units=50))
#model.add(Dense(units=1)) # add one fully connected layer

### Just for our records
#model.summary()

### Compile the model using base Keras stuff
#model.compile(optimizer="adam", loss="mean_squared_error")
#model.fit(X, y, epochs=200, batch_size=32)