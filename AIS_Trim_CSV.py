from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import extra stuff for class parsing and the like
import AIS_Path_Utils as utils

from math import radians, cos, sin, asin, sqrt
import sys
import os
sys.path.append("..")
#import utils
from datetime import datetime
import time
from io import StringIO
from pyproj import Geod
geod = Geod(ellps='WGS84')
#matplotlib inline

### Paths for input data
dataset_path = "/Volumes/PATRIOT/AIS Project/Marine Cadastre Datasets/CSVs/" 
save_path = "/Users/samyakovlev/Desktop/"

csvs_provided = ["AIS_2017_05_Zone15.csv", "AIS_2017_05_Zone16.csv"]

# save_file = "pickled_mc_dataset.pkl" #we don't care about pickles yet

### Starting parameters
### New Orleans
LAT_MIN = 27.5
LAT_MAX = 30.5
LON_MIN = -90.5
LON_MAX = -87.5
LAT_MIN, LAT_MAX, LON_MIN, LON_MAX = utils.get_boundaries("New Orleans")

# Maximum speed, in knots
SPEED_MAX = utils.SPEED_MAX

# Base time, for use in converting human-readable timestamps to seconds
EPOCH = datetime(1970, 1, 1)

# LAT, LON, SOG, COG, HEADING, ROT, NAV_STT, TIMESTAMP, MMSI = list(range(9)) # may not need this format yet?

#def trim(csv_to_trim = "AIS_2017_05_Zone15.csv"):

csv_list=[]

for csv_to_trim in csvs_provided:
	print("--------- Reading in: " + dataset_path + csv_to_trim)
	csv_list.append(pd.read_csv(dataset_path + csv_to_trim))

ais_data = pd.concat(csv_list, ignore_index=True)
#vessels_fishing		= ais_data[ais_data.VesselType.isin(utils.fishing)]	# fishing vessels subset for training/plotting elsewhere
#vessels_passenger 	= ais_data[ais_data.VesselType.isin(utils.passenger)]	# passenger vessels
#vessels_cargo 		= ais_data[ais_data.VesselType.isin(utils.cargo)]		# cargo vessels
#vessels_tanker 		= ais_data[ais_data.VesselType.isin(utils.tanker)]		# tanker vessels
#vessels_pleasure 	= ais_data[ais_data.VesselType.isin(utils.pleasurecraft)]	# pleasure craft
#vessels_speed 		= ais_data[ais_data.VesselType.isin(utils.speedy)]		# speedcraft, likely will not be used
#vessels_tug 		= ais_data[ais_data.VesselType.isin(utils.tug)]			# tugboats

print ("------------------------ 1. Removing extraneous columns ------------------------")
print(ais_data.info())

### Remove unneeded columns
columns_to_remove = ["VesselName", "IMO", "CallSign", "Status", "Length", "Width", "Draft", "Cargo"] #, "VesselType"]
ais_data.drop(columns_to_remove, inplace=True, axis=1)

print ("------------------------ After extra column removal ----------------------------")
print(ais_data.info())

### Get rid of items not within time range
#	This is one of the longest parts of the process
#	Whoever finds this, please find a faster way
print ("----------------------- 2.0 Converting Timestamps to Seconds -------------------")
#ais_data["BaseDateTime"] = ais_data["BaseDateTime"].apply(lambda x: ((datetime.strptime(x, "%Y-%m-%dT%H:%M:%S")) - EPOCH).total_seconds())
# The following method takes apparently 1/8th the time of datetime.strptime, which is HUGE
# ais_data["BaseDateTime"] = ais_data["BaseDateTime"].apply(lambda x: ((datetime(int(x[0:4]), int(x[5:7]), int(x[8:10]), int(x[11:13]), int(x[14:16]), int(x[17:19]))) - EPOCH).total_seconds())
ais_data["BaseDateTime"] = ais_data["BaseDateTime"].apply(lambda x: (datetime(int(x[:4]), int(x[5:7]), int(x[8:10]), int(x[11:13]), int(x[14:16]), int(x[17:19])) - EPOCH).total_seconds())
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

print ("----------------------- 4. Normalizing data for training ------------------------")
### Normalize all data between 0 and 1
#	This makes training go a little easier
ais_data.LAT = (ais_data.LAT - LAT_MIN)/(LAT_MAX-LAT_MIN) # normal LAT as defined by coordinates
ais_data.LON = (ais_data.LON - LON_MIN)/(LON_MAX-LON_MIN) # normal LON as defined by cooridnates
ais_data.COG /= 360.0 												# get COG between -1 and 1
ais_data.COG = ais_data.COG.apply(lambda x: x+1 if x < 0 else x) 	# add 1 to all negative COGs to make them coterminal with the original 
ais_data.SOG /= SPEED_MAX 	# normal speed
ais_data.Heading /= 360.0
ais_data.Heading = ais_data.Heading.apply(lambda x: x+1 if x <0 else x) # repeat as above

#print ("----------------------- 4.1 Cutting discontinuous paths -----------------------")
#paths = {}
#mmsi_groups = ais_data.groupby(ais_data.index) #"MMSI") # Split into all the MMSIs, remove MMSI column, sort by time, and reset index
#for MMSI in mmsi_groups.groups:
#	paths[MMSI] = ais_data[ais_data.index == MMSI].sort_values(by="BaseDateTime").reset_index(drop=True) #.drop(["MMSI"], axis=1).sort_values(by="BaseDateTime").reset_index(drop=True)

#utils.cut_discontinuous_tracks(paths, 2)

print ("--------------------- 5. Generating vessel class subframes ----------------------")
vessels_fishing		= ais_data[ais_data.VesselType.isin(utils.fishing)]		# fishing vessels 
vessels_passenger 	= ais_data[ais_data.VesselType.isin(utils.passenger)]	# passenger vessels
vessels_cargo 		= ais_data[ais_data.VesselType.isin(utils.cargo)]		# cargo vessels
vessels_tanker 		= ais_data[ais_data.VesselType.isin(utils.tanker)]		# tanker vessels
vessels_pleasure 	= ais_data[ais_data.VesselType.isin(utils.pleasurecraft)]	# pleasure craft
vessels_speed 		= ais_data[ais_data.VesselType.isin(utils.speedy)]		# speedcraft, likely will not be used
vessels_tug 		= ais_data[ais_data.VesselType.isin(utils.tug)]			# tugboats

print ("---------------------------- Writing final to .csv -----------------------------")
ais_data.to_csv(save_path + "trimmed_M5_Z15_ULTRAtrim.csv", index=False)
vessels_fishing.to_csv(save_path + "trimmed_fishing.csv", index=False)
vessels_passenger.to_csv(save_path + "trimmed_passenger.csv", index=False)
vessels_cargo.to_csv(save_path + "trimmed_cargo.csv", index=False)
vessels_tanker.to_csv(save_path + "trimmed_tanker.csv", index=False)
vessels_pleasure.to_csv(save_path + "trimmed_pleasurecraft.csv", index=False)
vessels_speed.to_csv(save_path + "trimmed_speedcraft.csv", index=False)
vessels_tug.to_csv(save_path + "trimmed_tug.csv", index=False)
#def __init__():
#	trim()