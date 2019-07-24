
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from haversine import haversine, Unit
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM
from argparse import ArgumentParser

import AIS_Path_Utils as utils

#New York 
LAT_MIN = 40.4
LAT_MAX = 40.8
LON_MIN = -74.3
LON_MAX = -73.5

#--------------------------------------------------------------
### Collects data from given csv and assigns training and test data
def split_data(data):
    
    mmsi_groups = data.groupby("MMSI")     #groups data into individual tracks
    
    vTypes=[0, 0, 0, 0, 0, 0, 0]

    count = 0
    
    ### loop through each track
    for MMSI in mmsi_groups.groups:
    
        print(len(mmsi_groups.groups))
        print("Vessel #" +  str(count))
        count+=1
        
        path = data[data.MMSI == MMSI].sort_values(by="BaseDateTime").reset_index(drop=True) 
        #utils.cut_discontinuous_tracks([path], 2) need to rework either this or utils to allow for singular paths
    
        input_feature = path.iloc[:,[1,2,3,4,5]].values     #input variables (LAT,LON,SOG,COG,Heading)
        input_data = input_feature
        
        ### plot path
        #plt.plot(input_feature[:,1], input_feature[:,0])
        #plt.xlabel("Longitude")
        #plt.ylabel("Latitude")
        #plt.show() 
    
        ### choose input and output train data
        lookback = 30     #the number of past data points used for prediction
        X=[]               #Train inputs
        Y=[]               #Train outputs
        for i in range(len(path)-lookback-1):
            t=[]
            for j in range(0,lookback):
                t.append(input_data[(i+j), :])
            X.append(t)
            Y.append(input_data[i+ lookback, :])
        
        ### reshape into proper format (samples, time step, #features)    
        X, Y= np.array(X), np.array(Y)
        X = X.reshape(X.shape[0],lookback,5)
        Y = Y.reshape(Y.shape[0],5)
    
        print(X.shape)
        print(Y.shape)
    
        if(count < 2 * int(len(mmsi_groups.groups)/3)):    #choose number of training vessels
            trainX.append(X)
            trainY.append(Y)
             vType = utils.resolve_vessel_type(path.iloc[0, 6]) # get vesseltype
            if vType == "Fishing":
                vTypes[0] += 1
            elif vType == "Cargo":
                vTypes[1] += 1
            elif vType == "Tanker":
                vTypes[2] += 1
            elif vType == "Pleasurecraft":
                vTypes[3] += 1
            elif vType == "High-speed":
                vTypes[4] += 1
            elif vType == "Passenger":
                vTypes[5] += 1
            else:
                vTypes[6] += 1
        elif(count < len(mmsi_groups.groups)):  #choose number of test vessels
            testX.append(X)
            testY.append(Y)
        else:
            break
    print("Fishing vessels: " + str(vTypes[0]))
    print("Cargo vessels: " + str(vTypes[1]))
    print("Tanker vessels: " + str(vTypes[2]))
    print("Pleasure vessels: " + str(vTypes[3]))
    print("High-speed vessels: " + str(vTypes[4]))
    print("Passenger vessels: " + str(vTypes[5]))
    print("Other: " + str(vTypes[6]))  
    return trainX,trainY,testX,testY
#-----------------------------------------------------------------------------------------
### Train the model given inputs and outputs
def train(model,trainX,trainY, num_epochs):
    print("preparing to train")   
    
    for i in range(len(trainX)):
        print("Train Vessel #" +  str(i))
        inputs = trainX[i]
        outputs = trainY[i]
    
        try:
            model.fit(inputs, outputs, epochs=num_epochs, batch_size=inputs.shape[0])
        except:
            print("untrainable: moving on")
            continue
            
    return model
#--------------------------------------------------------------------------------------------
### Make and compare predictions based on test data 
def predict(model,testX,testY):
    
    predictions = []

    for i in range(len(testX)):
        print("Test Vessel #" + str(i))
        inputs = testX[i]
        outputs = testY[i]
    
        predictions.append(model.predict(inputs))
    
    return predictions
     
#------------------------------------------------------------------------------------------
### Helper for plot predictions. Plots a single prediction and actual path 
def plot_predictions_help(prediction, actual):
    
    predicted_norm_LON = []
    predicted_norm_LAT = []
    predicted_anom_LON = []
    predicted_anom_LAT = []
    actual_norm_LON = []
    actual_norm_LAT = []
    actual_anom_LON = []
    actual_anom_LAT = []
    
    paths = []
    
    anom_count = 0
    
    #Calculate prediction errors
    for i in range(len(prediction)):
        
        #re scale lat and lon
        pred_lat = prediction[i,0] * (LAT_MAX-LAT_MIN) + LAT_MIN
        pred_lon = prediction[i,1] * (LON_MAX-LON_MIN) + LON_MIN
        actual_lat = actual[i,0] * (LAT_MAX-LAT_MIN) + LAT_MIN
        actual_lon = actual[i,1] * (LON_MAX-LON_MIN) + LON_MIN
        
        distance = haversine((pred_lat, pred_lon), (actual_lat, actual_lon), unit=Unit.NAUTICAL_MILES)
        
        LAT_error = abs(prediction[i,0] - actual[i,0])
        LON_error = abs(prediction[i,1] - actual[i,1])
        #SOG_error = abs(predictions[i,2] - outputs[i,2])
        #COG_error = abs(predictions[i,3] - outputs[i,3])

        if(LAT_error > .025 and LON_error > 0.025 ): #adjust this if for chosen error function
            actual_anom_LON.append(actual_lon)
            actual_anom_LAT.append(actual_lat)
            predicted_anom_LON.append(pred_lon)
            predicted_anom_LAT.append(pred_lat)
            anom_count +=1

        else:
            actual_norm_LON.append(actual_lon)
            actual_norm_LAT.append(actual_lat)
            predicted_norm_LON.append(pred_lon)
            predicted_norm_LAT.append(pred_lat)
        
    #plt.figure()   
    plt.scatter(predicted_norm_LON,predicted_norm_LAT, marker = 'x', s = 10, color = 'blue', label = 'Predicted Position')
    plt.scatter(predicted_anom_LON,predicted_anom_LAT, marker = 'x', s = 10, color = 'red', label = 'Predicted Anomaly')
    plt.scatter(actual_norm_LON,actual_norm_LAT, marker = 'o', s = 10, color = 'green',label =  "Actual Position")
    plt.scatter(actual_anom_LON,actual_anom_LAT, marker = 'o', s = 10, color = 'red', label =  "Anomalous Position")

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()

    #print("Saving figure")

    plt.savefig("/Users/samyakovlev/Desktop/RNN_Results/" + args.type + str(i) + ".png", bbox_inches = "tight", dpi = 200)
    plt.clf()   # Clear figure, for next figure
    #plt.show()
    
    if(anom_count > len(prediction) / 3):
        paths.append([[actual_norm_LON + actual_anom_LON, actual_norm_LAT + actual_anom_LAT], True])
    else:
        paths.append([[actual_norm_LON + actual_anom_LON, actual_norm_LAT + actual_anom_LAT], False])      

#------------------------------------------------------------------------------------------------------------
#   Plots individual graphs of predictions and corresponding actual path. Calls helper method to plot one at a time
def plot_predictions(predictions, actuals):
    for i in range(len(predictions)):
        plot_predictions_help(predictions[i], actuals[i])
#----------------------------------------------------------------------------------------------------------
def plot_over_map(paths):
    
    plt.ylim(top=LAT_MAX, bottom=LAT_MIN)
    plt.xlim(left=LON_MIN, right=LON_MAX)
    
    im = plt.imread(utils.current_region+".png")
    #im = plt.imread("/Users/Eamon/Desktop/MSC/SRI-2019-AIS-Anomaly-Detection/New York.png") #plot map of port as background
    plt.imshow(im, extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX])
    
    for x in paths:
        path = x[0]
        anom_status = x[1]
            
        if(anom_status == True):
            plt.plot(path[0],path[1], lw = 0.6, color = 'red')
        else:
            plt.plot(path[0],path[1], lw = 0.6, color = 'green')
    
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.savefig("/Users/Eamon/Desktop/realtimepics/map.png", bbox_inches = "tight", dpi = 200)
    plt.show()
#--------------------------------------------------------------------------------------------

### MAIN
#   Include command line arguments for training on different files
parser = ArgumentParser()
parser.add_argument('action', type=str, default="Test", help="Train or Test for given vessel type")
parser.add_argument('type', type=str, help="Vessel type to be used for training/testing\nFiles must be named \'trimmed_[VESSEL TYPE].csv\'")
#   Add file location to make inputting filenames simpler
#   IF YOU ARE USING COMMAND LINE ARGS, CHANGE THIS
dataset_path = "/Users/samyakovlev/Desktop/SRI-2019-AIS-Anomaly-Detection/"
args = parser.parse_args()

trainX = []
trainY = []
testX = []
testY = []

### Import data
#   Done if the network is told to Train
if args.action.lower() == "train":
    data = pd.read_csv(dataset_path + "trimmed_" + args.type + ".csv", header=0, index_col=0)
    trainX,trainY,testX,testY = split_data(data)   #collect data

    ##build model
    model = Sequential()
    model.add(LSTM(units=100, return_sequences= True, input_shape=(None,5)))
    #model.add(LSTM(units=30, return_sequences=True))
    model.add(LSTM(units=100))
    model.add(Dense(units=5 ))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics = ['accuracy'])

    trained_model = train(model,trainX,trainY, 20)   #train model

    model.save("/Users/samyakovlev/Desktop/"+args.type+"_model.h5")   #save model per vessel type

elif args.action.lower() == "test":
    model = load_model("/Users/samyakovlev/Desktop/"+args.type+"_model.h5")
    model.summary()                                 # Quick output to show the model was loaded

    data = pd.read_csv(dataset_path + "trimmed_" + args.type + ".csv", header=0, index_col=0)
    trainX,trainY,testX,testY = split_data(data)    # collect data
    all_predictions = predict(model,testX,testY)    # make predictions
    plot_predictions(all_predictions, testY)        # plot predictions

else:
    print("Please specify whether to TRAIN or TEST on the given vessel type")
