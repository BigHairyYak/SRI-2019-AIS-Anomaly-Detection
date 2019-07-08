from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from math import radians, cos, sin, asin, sqrt
import sys
import os
sys.path.append("..")
import utils
import pickle
import copy
from datetime import datetime
import time
from io import StringIO
from pyproj import Geod
geod = Geod(ellps='WGS84')
#matplotlib inline

### Paths for input data
dataset_path = "/Volumes/PATRIOT/AIS Project/Marine Cadastre Datasets/CSVs/" 
save_path = "/Users/samyakovlev/Desktop/"

csv_to_trim = "AIS_2017_05_Zone15.csv"

# save_file = "pickled_mc_dataset.pkl" #we don't care about pickles yet

### Starting parameters
### New Orleans
LAT_MIN = 27.5
LAT_MAX = 30.5
LON_MIN = -90.5
LON_MAX = -87.5

# Maximum speed, in knots
SPEED_MAX = 30.0

# Base time, for use in converting human-readable timestamps to seconds
EPOCH = datetime(1970, 1, 1)

# LAT, LON, SOG, COG, HEADING, ROT, NAV_STT, TIMESTAMP, MMSI = list(range(9)) # may not need this format yet?

#def trim(csv_to_trim = "AIS_2017_05_Zone15.csv"):

print("--------- Reading in: " + dataset_path + csv_to_trim)
ais_data = pd.read_csv(dataset_path + csv_to_trim)
vessels_passenger = pd.DataFrame()	# for auxillary dataset for only passenger vessels
vessels_cargo = pd.DataFrame()		# for auxillary dataset for only cargo/freight vessels
vessels_tanker = pd.DataFrame()

print ("------------------------ 1. Removing extraneous columns ------------------------")
print(ais_data.info())

### Remove unneeded columns
columns_to_remove = ["VesselName", "IMO", "CallSign", "Status", "Length", "Width", "Draft", "Cargo"]
ais_data.drop(columns_to_remove, inplace=True, axis=1)

print ("------------------------ After extra column removal ----------------------------")
print(ais_data.info())

### Get rid of items not within time range
print ("----------------------- 2.0 Converting Timestamps to Seconds ------------------")
ais_data["BaseDateTime"] = ais_data["BaseDateTime"].apply(lambda x: ((datetime.strptime(x, "%Y-%m-%dT%H:%M:%S")) - EPOCH).total_seconds())
t_min = time.mktime(time.strptime("31/05/2017 23:59:59", "%d/%m/%Y %H:%M:%S"))
t_max = time.mktime(time.strptime("31/08/2017 23:59:59", "%d/%m/%Y %H:%M:%S"))
# I don't think this is necessary what with all the time format conversions going on
# something something time here


### Removing immediately unusable items, as defined by given coordinates
print ("----------------------- 2.1 Removing out-of-bounds items -----------------------")
ais_data = ais_data[ais_data.LAT >= LAT_MIN]
ais_data = ais_data[ais_data.LAT <= LAT_MAX]
ais_data = ais_data[ais_data.LON >= LON_MIN]
ais_data = ais_data[ais_data.LON <= LON_MAX]

print ("----------------------- 2.2 Removing strange speed items -----------------------")
ais_data = ais_data[ais_data.SOG > 0]
ais_data = ais_data[ais_data.SOG < SPEED_MAX]

print ("--------------------------- After out-of-bounds removal ------------------------")
print(ais_data.info())

print ("----------------------- 3.1 Sorting by MMSI Values -----------------------------")
ais_data = ais_data.sort_values(by='MMSI')

### Sort of checkpoint
#ais_data.to_csv(save_path + "AIS_M5_Z15.csv", index=False)

### Group all items with the same MMSI into groups - Everything to this point is confirmed working
### Maybe do this in actual training phase?
print ("----------------------- 3.2 Grouping MMSI Values... ----------------------------")
ais_paths = ais_data.groupby(ais_data.MMSI)
print ("Group sizes:")
print(ais_paths.size())

print ("----------------------- 3.3 Removing short (<20) paths -------------------------")
ais_data = ais_paths.filter(lambda x: len(x) >= 20)

### Write to .csv for further processing in numpy or something
#filtered_paths.to_csv(save_path + "trimmed_M5_Z15.csv", index=False)
ais_paths = ais_data.groupby(ais_data.MMSI)

print ("----------------------- 3.4 Removing tracks with no heading --------------------")
# Heading values of 511.0 mean default, or unavailable
ais_data = ais_data[ais_data.Heading != 511.0]
#ais_paths = ais_paths.filter(lambda path: len(path) >= 20).groupby(ais_paths.MMSI)
#print("Group sizes:")
#print(ais_paths.size())
#paths_list = list(paths) # done for iteration purposes or something

#print ("----------------------- 3.4 Arranging tracks in paths by time -----------------")
# This is done in AIS_RNN_Yakovlev.py, upon loading of pre-processed .csv
# Every MMSI group is pulled out as its own DataFrame, which in turn is sorted

#print ("----------------------- 3.5 Removing long (>4hr) paths ------------------------")
# Something here to look for max timestamp and min timestamp, subtract and see if it needs deleting

print ("----------------------- 4 Normalizing data for training ------------------------")
ais_data.LAT = (ais_data.LAT - LAT_MIN)/(LAT_MAX-LAT_MIN) # normal LAT as defined by coordinates
ais_data.LON = (ais_data.LON - LON_MIN)/(LON_MAX-LON_MIN) # normal LON as defined by cooridnates
ais_data.COG /= 360.0 		# normal COG between -1 and 1
ais_data.SOG /= SPEED_MAX 	# normal speed

ais_data.Heading /= 360.0

print ("------------------------------- Writing final to .csv --------------------------------")
ais_data.to_csv(save_path + "trimmed_M5_Z15.csv", index=False)

#def __init__():
#	trim()