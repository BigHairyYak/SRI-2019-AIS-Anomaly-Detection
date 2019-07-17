##This program loads a previously trained model and runs predictions on given data
##to be used in addition to AIS_RNN.py 
##works best in jupyter notebook (run code from AIS_RNN.py first for training, 
    #then in same notebook run this code for predictions)
import math
from haversine import haversine, Unit
from tensorflow.keras.models import load_model
model = load_model('/Users/Eamon/Desktop/AIS_RNN2.h5')

data = pd.read_csv("/Users/Eamon/Desktop/test data/allNYTrimmed.csv", header=0, index_col=0) 
mmsi_groups = data.groupby(data.index)     #groups data into individual tracks


testX = []
testY = []

count = 0

###loop through each track
for MMSI in mmsi_groups.groups:
    
    print(len(mmsi_groups.groups))
    
    print("Vessel #" +  str(count))
    count+=1
        
    path = data[data.index == MMSI].sort_values(by="TSTAMP").reset_index(drop=True) #TSTAMP for AISHub Data, #BaseDateTime for Marine Cadestre
    
    input_feature = path.iloc[:,[1,2,3,4]].values     #input variables (LAT,LON,SOG,COG)
    input_data = input_feature
    
    
    ###plot path
    plt.plot(input_feature[:,0], input_feature[:,1])
    plt.xlabel("Latitude")
    plt.ylabel("Longitude")
    plt.show() 
    
    ###choose input and output train data
    lookback = 30     #the number of past data points used for prediction
    X=[]               #Train inputs
    Y=[]               #Train outputs
    
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
    
    testX.append(X)
    testY.append(Y)

        
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
    

    LAT_MIN = 40.4
    LAT_MAX = 40.8
    LON_MIN = -74.3
    LON_MAX = -73.5
    
    
    #Calculate prediction errors
    for i in range(len(predictions)):
        
        #re scale lat and lon
        pred_lat = predictions[i,0] * (LAT_MAX-LAT_MIN) + LAT_MIN
        pred_lon = predictions[i,1] * (LON_MAX-LON_MIN) + LON_MIN
        actual_lat = outputs[i,0] * (LAT_MAX-LAT_MIN) + LAT_MIN
        actual_lon = outputs[i,1] * (LON_MAX-LON_MIN) + LON_MIN
        
        distance = haversine((pred_lat, pred_lon), (actual_lat, actual_lon), unit=Unit.NAUTICAL_MILES)
 
        
        LAT_error = abs(predictions[i,0] - outputs[i,0])
        LON_error = abs(predictions[i,1] - outputs[i,1])
        #SOG_error = abs(predictions[i,2] - outputs[i,2])
        #COG_error = abs(predictions[i,3] - outputs[i,3])
        


        if(LAT_error > .1 or LON_error > 0.1 or (LAT_error > .05 and LON_error > .05)) : #or SOG_error > .03 or COG_error > .03):  #Error function
            plt.plot(predictions[i,0], predictions[i,1],'x',markersize=5, color= 'red')   
            plt.plot(outputs[i, 0], outputs[i,1], 'o',markersize=5,  color='red')
        else:
            pass
            plt.plot(predictions[i,0], predictions[i,1],'x',markersize=3, color= 'blue')   
            plt.plot(outputs[i, 0], outputs[i,1], 'o',markersize=3,  color='green')
            
    plt.show()