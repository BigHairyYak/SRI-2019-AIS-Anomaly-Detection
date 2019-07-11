import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import math
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM


#import data for each vessel

###build model
model = Sequential()
model.add(LSTM(units=30, return_sequences= True, input_shape=(None,4)))
model.add(LSTM(units=30, return_sequences=True))
model.add(LSTM(units=30))
model.add(Dense(units=4))
model.compile(optimizer='adam', loss='mean_squared_error', metrics = ['accuracy'])


data = pd.read_csv("/Users/Eamon/Desktop/cargo.csv", header=0, index_col=0) 
mmsi_groups = data.groupby(data.index)     #groups data into individual tracks

count = 0
testPath = None

###loop through each track
for MMSI in mmsi_groups.groups:
    print("hey: " +  str(count))
    count+=1
    
    if(count == 51):     #train on first x-1 tracks, use xth for testing
        testPath = path = data[data.index == MMSI].sort_values(by="BaseDateTime").reset_index(drop=True)
        break
        
    path = data[data.index == MMSI].sort_values(by="BaseDateTime").reset_index(drop=True) 
    
    input_feature = path.iloc[:,[1,2,3,4]].values     #input variables (LAT,LON,SOG,COG)
    input_data = input_feature
    
    
    ###plot path
    plt.plot(input_feature[:,0], input_feature[:,1])
    plt.xlabel("Latitude")
    plt.ylabel("Longitude")
    plt.show()
    
    ###normalize data
    sc= MinMaxScaler(feature_range=(0,1))
    input_data[:,0:4] = sc.fit_transform(input_feature[:,:])
    
    
    ###choose input and output train data
    lookback = 100     #the number of past data points used for prediction
    X=[]               #Train inputs
    Y=[]               #Train outputs
    
    for i in range(len(path)-lookback-1):
        t=[]
        for j in range(0,lookback):
            t.append(input_data[[(i+j)], :])
        X.append(t)
        Y.append(input_data[i+ lookback, :])
        
    #reshape into proper format (samples, time step, #features)    
    X, Y= np.array(X), np.array(Y)
    X = X.reshape(X.shape[0],lookback,4)
    Y = Y.reshape(Y.shape[0],4)
    
    print(X.shape)
    print(Y.shape)

    ###train
    try:
        model.fit(X, Y, epochs=100, batch_size=X.shape[0])
    except:
        print("untrainable: moving on")
        continue
    #model.reset_states()
    

###Gather test data
input_feature = testPath.iloc[:,[1,2,3,4]].values
input_data = input_feature
sc= MinMaxScaler(feature_range=(0,1))
input_data[:,0:4] = sc.fit_transform(input_feature[:,:])

lookback= 100

test_size=int(.3 * len(testPath))
X_test=[]

for i in range(len(testPath)-lookback-1):
    t=[]
    for j in range(0,lookback):
        t.append(input_data[[(i+j)], :])
    X_test.append(t)

        
X_test =  np.array(X_test)
X_test = X_test.reshape(X_test.shape[0],lookback,4)
X_test = X_test[:len(X_test)//2] #first half of path - this way it predicts second half
Y_test = input_data[(lookback): X_test.shape[0] + lookback, :]
Y_test = Y_test.reshape(Y_test.shape[0],4)

###predict
predictions = model.predict(X_test)

print(X_test.shape)
print(predictions.shape)



###plot predictions
#Plot Path (Lat vs Lon)
plt.plot(predictions[:,0], predictions[:,1], color= 'red')   
plt.plot(Y_test[:, 0], Y_test[:,1],  color='green')
plt.show()

#Plot LAT
plt.plot(predictions[:,0], color= 'red')
plt.plot(Y_test[:,0], color='green')
plt.show()

#Plot LON
plt.plot(predictions[:,1], color= 'red')
#plt.plot(input_data[lookback:test_size+(2*lookback), 1 ], color='green')
plt.plot(Y_test[:,1], color='green')
plt.show()

#Plot SOG
plt.plot(predictions[:,2], color= 'red')
#plt.plot(input_data[lookback:test_size+(2*lookback), 2 ], color='green')
plt.plot(Y_test[:,2], color='green')
plt.show()

#Plot COG
plt.plot(predictions[:,3], color= 'red')
#plt.plot(input_data[lookback:test_size+(2*lookback), 3 ], color='green')
plt.plot(Y_test[:,3], color='green')
plt.show()