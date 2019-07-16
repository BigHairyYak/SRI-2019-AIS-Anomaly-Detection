import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM




###build model
model = Sequential()
model.add(LSTM(units=100, return_sequences= True, input_shape=(None,4)))
#model.add(LSTM(units=30, return_sequences=True))
model.add(LSTM(units=100))
model.add(Dense(units=4))
model.compile(optimizer='adam', loss='mean_squared_error', metrics = ['accuracy'])

#import data
data = pd.read_csv("/Users/Eamon/Desktop/cargo.csv", header=0, index_col=0) 
mmsi_groups = data.groupby(data.index)     #groups data into individual tracks

count = 0
testPath = None

trainX = []
trainY = []
testX = []
testY = []

###loop through each track and assign data to train and test groups
for MMSI in mmsi_groups.groups:
    
    print(len(mmsi_groups.groups))
    
    print("Vessel #" +  str(count))
    count+=1
        
    path = data[data.index == MMSI].sort_values(by="BaseDateTime").reset_index(drop=True) 
    
    input_feature = path.iloc[:,[1,2,3,4]].values     #input variables (LAT,LON,SOG,COG)
    input_data = input_feature
    
    
    ###plot path
    plt.plot(input_feature[:,0], input_feature[:,1])
    plt.xlabel("Latitude")
    plt.ylabel("Longitude")
    plt.show() 
    
    ###choose input and output train data
    lookback = 30     #the number of past data points used for prediction
    X=[]               #inputs
    Y=[]               #outputs
    
    for i in range(len(path)-lookback-1):
        t=[]
        for j in range(0,lookback):
            t.append(input_data[(i+j), :])
        X.append(t)
        Y.append(input_data[i+ lookback, :])
        
    #reshape into proper format (samples, time step, #features)    
    X, Y= np.array(X), np.array(Y)
    X = X.reshape(X.shape[0],lookback,4)
    Y = Y.reshape(Y.shape[0],4)
    
    print(X.shape)
    print(Y.shape)
    
    
    if(count < 101):    #choose number of training vessels
        trainX.append(X)
        trainY.append(Y)
    elif(count < 201):  #choose number of test vessels
        testX.append(X)
        testY.append(Y)
    else:
        break
        
    
###Train
for i in range(len(trainX)):
    print("Train Vessel #" +  str(i))
    inputs = trainX[i]
    outputs = trainY[i]
    
    try:
        model.fit(inputs, outputs, epochs=100, batch_size=inputs.shape[0])
    except:
        print("untrainable: moving on")
        continue
        
###Predict
for i in range(len(testX)):
    print("Test Vessel #" + str(i))
    inputs = testX[i]
    outputs = testY[i]
    
    predictions = model.predict(inputs)
    
    try:     #for unusable tracks (empty inputs)
        print(predictions.shape)
    except:
        continue
        
    print(outputs.shape)
    print(inputs.shape)
    
    
    #Calculate prediction errors
    for i in range(len(predictions)):
        LAT_error = abs(predictions[i,0] - outputs[i,0])
        
        LON_error = abs(predictions[i,1] - outputs[i,1])
        
        SOG_error = abs(predictions[i,2] - outputs[i,2])
        
        COG_error = abs(predictions[i,3] - outputs[i,3])
        
        if(LAT_error > .03 and LON_error > .03 or SOG_error > .03 or COG_error > .03):  #Error function
            plt.plot(predictions[i,0], predictions[i,1],'o',markersize=1, color= 'blue')   
            plt.plot(outputs[i, 0], outputs[i,1], 'o',markersize=1,  color='blue')
        else:
            plt.plot(predictions[i,0], predictions[i,1],'o',markersize=1, color= 'red')   
            plt.plot(outputs[i, 0], outputs[i,1], 'o',markersize=1,  color='green')
            
    plt.show()