import math
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.widgets
from haversine import haversine, Unit
### Global Variables for use 
#	Vessel Codes
fishing = [30, 1001, 1002]
passenger = [60, 61, 62, 63, 64, 69]
cargo = [70, 71, 72, 73, 74, 79, 1003,1004,1016]
tanker = [80, 81, 82, 83, 84, 89, 1024]
pleasurecraft = [36, 37, 1019]
speedy = [40, 41, 42, 43, 44, 49]
tug = [31, 32, 52, 1025]

# Maximum Speed (knots)
# 1 knot = 1 nautical mile per hour
SPEED_MAX = 30.0

# Radius of the Earth (in nautical miles)
R_EARTH = 3443.9

### Different Port Coordinates
#	Set in terms of LAT_MIN, LAT_MAX, LON_MIN, LON_MAX
regions = dict()
regions.update({"New Orleans" : [27.5, 30.5, -90.5, -87.5]})
regions.update({"Los Angeles" : [31.5, 34.5, -121.5, -117.5]})
regions.update({"New York" : [40.4, 40.8, -74.3, -73.5]})
regions.update({"Houston" : [27.75, 30, -97.5, -93]})
current_region = ""

LAT_MIN, LAT_MAX, LON_MIN, LON_MAX = regions["New Orleans"] #= range(4)

def get_boundaries(region="New Orleans"):
	global current_region # allows us to modify current_region
	current_region = region
	global LAT_MIN, LAT_MAX, LON_MIN, LON_MAX
	global bounds
	bounds = regions[current_region]	# Update local bounds
	LAT_MIN = bounds[0]
	LAT_MAX = bounds[1]
	LON_MIN = bounds[2]
	LON_MAX = bounds[3]
	return bounds

# Establish default values of boundaries, if ever used
bounds = get_boundaries()

### Return vessel types as a string based on code in AIS data
#	This is pretty self-explanatory
def resolve_vessel_type(vessel_code):
	try:
		vType = int(vessel_code)
	except:
		return("FAILED") # if number can not be converted, return failure
	if vType in fishing:
		return("Fishing")
	elif vType in cargo:
		return("Cargo")
	elif vType in tanker:
		return("Tanker")
	elif vType in tug:
		return("Tug")
	elif vType in pleasurecraft:
		return("Pleasurecraft")
	elif vType in speedy:
		return("High-speed")
	elif vType in passenger:
		return("Passenger")
	else:
		return("Unknown") # if nothing is found, report as such



################################## PLOTTING FUNCTIONS ##################################

### Plot vessel courses over a map, to help visualize
#	First establish boundaries to plot between
#	Then show image, then plot the requested number of vessels
def plot_over_map(paths = {}, num_vessels = 20, show_map=True, label_MMSI=False, sort_by_type=False, save_image=False):
	plt.title("Vessels in the " + current_region + " area")
	plt.ylim(top=bounds[1], bottom=bounds[0])
	plt.xlim(left=bounds[2], right=bounds[3])
	plt.xlabel("Longitude")
	plt.ylabel("Latitude")
	if show_map:
		im = plt.imread(current_region + ".png")
		plt.imshow(im, extent=[bounds[2],bounds[3], bounds[0], bounds[1]])
	path_iterator = iter(paths)
	path_color = "b"
	for i in range(num_vessels):
		path = next(path_iterator)
		if sort_by_type:
			#print(str(paths[path].VesselType[0]) + ": " + resolve_vessel_type(paths[path].VesselType[0]))
			if resolve_vessel_type(paths[path].VesselType[0]) == "Fishing":
				path_color = "b"
			elif resolve_vessel_type(paths[path].VesselType[0]) == "Cargo":
				path_color = "#FF0033" #"r"
			elif resolve_vessel_type(paths[path].VesselType[0]) == "Tanker":
				path_color = "#cb4335" #"m"
			elif resolve_vessel_type(paths[path].VesselType[0]) == "Tug":
				path_color = "#993333" #"b"
			elif resolve_vessel_type(paths[path].VesselType[0]) == "Pleasure":
				path_color = "#9933CC" #"g"
			elif resolve_vessel_type(paths[path].VesselType[0]) == "Passenger":
				path_color = "c"
			#elif resolve_vessel_type(paths[path].VesselType[0]) == "Unknown":
			else:
				path_color = "#336666" #matplotlib.colors.BASE_COLORS[] #"y"
			plt.plot(paths[path].LON * (LON_MAX-LON_MIN) + LON_MIN, paths[path].LAT * (LAT_MAX-LAT_MIN) + LAT_MIN, label=path, color = path_color, lw=0.6)
		else:
			plt.plot(paths[path].LON * (LON_MAX-LON_MIN) + LON_MIN, paths[path].LAT * (LAT_MAX-LAT_MIN) + LAT_MIN, label=path, lw=0.6, color = path_color)
		if (label_MMSI):
			plt.text(paths[path].LON.iloc[-1] * (LON_MAX-LON_MIN) + LON_MIN, paths[path].LAT.iloc[-1] * (LAT_MAX-LAT_MIN) + LAT_MIN, path)
	if save_image:
		plt.savefig("Vessels_general_"+current_region+".png", bbox_inches="tight", dpi=200)
	plt.show()

### Plot over a map, all vessels of a type
#	Provide paths, the number of vessels to look through, and the type
#	This will look THROUGH that number of vessels, not FOR that number of vessels
#	Lazy for now, but this is more to see the change in appearance when we look at specific types within a larger set
def plot_over_map_by_type(paths = {}, num_vessels=20, type = "Cargo", show_map=True, label_MMSI=False, save_image=False):
	plt.title("Plotting " + type + " Vessels in " + current_region)
	if show_map:
		im = plt.imread(current_region + ".png")
		plt.imshow(im, extent=[bounds[2],bounds[3], bounds[0], bounds[1]])
	plt.xlabel("Longitude")
	plt.ylabel("Latitude")
	path_iterator = iter(paths)
	for i in range(num_vessels):
		path = next(path_iterator)
		if (resolve_vessel_type(paths[path].VesselType[0]) == type):
			plt.plot(paths[path].LON * (LON_MAX-LON_MIN) + LON_MIN, paths[path].LAT * (LAT_MAX-LAT_MIN) + LAT_MIN, label=path, alpha=0.75, lw=0.5, color="b")
			if (label_MMSI):
				plt.text(paths[path].LON.iloc[0] * (LON_MAX-LON_MIN) + LON_MIN, paths[path].LAT.iloc[0] * (LAT_MAX-LAT_MIN) + LAT_MIN, path)
				plt.text(paths[path].LON.iloc[int(len(paths[path].LON)/2)] * (LON_MAX-LON_MIN) + LON_MIN, paths[path].LAT.iloc[int(len(paths[path].LON)/2)] * (LAT_MAX-LAT_MIN) + LAT_MIN, path)
				plt.text(paths[path].LON.iloc[-1] * (LON_MAX-LON_MIN) + LON_MIN, paths[path].LAT.iloc[-1] * (LAT_MAX-LAT_MIN) + LAT_MIN, path)
	if save_image:
		plt.savefig("Vessels_"+type+"_"+current_region+".png", bbox_inches="tight", dpi=200)
	plt.show()

### Plot tracks of vessels, for visualization purposes and understanding it with human eyes
#	Every graph is also labeled with its REPORTED vessel type
def plot_vessel_tracks(paths):
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
			plt.title(resolve_vessel_type(values[0, 6])) 
		else:
			plt.title(str(int(values[0, 6])) + ": " + resolve_vessel_type(values[0, 6]))
	plt.show() # Show off the first 25 ship tracks


################################## DATAFRAME FUNCTIONS ##################################
#	Some of the longer tracks to test on:
#	53003297 - 9623 tracks
#	210509000 - 6029 tracks
#	212301000 - 8084 tracks

### Calculate distance traveled over Earth (taking into account curvature of the earth)
#	Takes in starting LAT and LON, end LAT and LON
#	Adapted from: https://www.ridgesolutions.ie/index.php/2013/11/14/algorithm-to-calculate-speed-from-two-gps-latitude-and-longitude-points-and-time-difference/
#	Formula for 'distance' taken from http://jan.ucc.nau.edu/~cvm/latlon_formula.html
def distance_over_earth(lat1, lon1, lat2, lon2):
	# Start by denormalizing coordinates, and converting them to radians for calculation
	lat1 = (lat1 * (LAT_MAX-LAT_MIN) + LAT_MIN) * math.pi / 180.0
	lat2 = (lat2 * (LAT_MAX-LAT_MIN) + LAT_MIN) * math.pi / 180.0
	lon1 = (lon1 * (LON_MAX-LON_MIN) + LON_MIN) * math.pi / 180.0
	lon2 = (lon2 * (LON_MAX-LON_MIN) + LON_MIN) * math.pi / 180.0
	'''
	rho1 = R_EARTH * math.cos(lat1)
	z1 = R_EARTH * math.sin(lat1)
	x1 = rho1 * math.cos(lon1)
	y1 = rho1 * math.sin(lon1)

	rho2 = R_EARTH * math.cos(lat2)
	z2 = R_EARTH * math.sin(lat2)
	x2 = rho2 * math.cos(lon2)
	y2 = rho2 * math.sin(lon2)

	# Take the dot product of the above variables
	dot = (x1 * x2 + y1 * y2 + z1 * z2)
	cos_theta = dot / (R_EARTH * R_EARTH)

	lat1 = math.radians(abs(lat1 * (LAT_MAX-LAT_MIN) + LAT_MIN))# * math.pi / 180.0
	lat2 = math.radians(abs(lat2 * (LAT_MAX-LAT_MIN) + LAT_MIN))# * math.pi / 180.0
	lon1 = math.radians(abs(lon1 * (LON_MAX-LON_MIN) + LON_MIN))# * math.pi / 180.0
	lon2 = math.radians(abs(lon2 * (LON_MAX-LON_MIN) + LON_MIN))# * math.pi / 180.0
	#print("Calculating (" + str(lat1) + ", " + str(lon1) + ") to (" + str(lat2) + ", " + str(lon2) +")")
	x = math.cos(lat1)*math.cos(lon1)*math.cos(lat2)*math.cos(lon2) + math.cos(lat1)*math.sin(lon1)*math.cos(lat2)*math.sin(lon2) + math.sin(lat1)*math.sin(lat2) # * R_EARTH
	#distance = math.atan2(math.sqrt(1 - x*x), x)
	'''
	distance = haversine((lat1, lon1), (lat2, lon2), unit=Unit.NAUTICAL_MILES)

	# Distance in nautical miles
	return distance #R_EARTH * cos_theta

### Remove unusual tracks from a given dataframe
#	An AIS message is considered unusual/anomalous if the speed is unreasonable, e.g.:
#	- The reported speed is greater than SPEED_MAX
#	- The calculated speed is greater than SPEED_MAX
#	Dataframes used for this AIS project have the structure [BaseDateTime, LAT, LON, SOG, COG, Heading, VesselType]
def plot_stop_go(paths):
	# this was originally based on some geod-related math and matrix manipulation
	# Now done using haversine formula, which supposedly works
	path_iterator = iter(paths)
	for i in range(100):
		next_path = next(path_iterator)
		stop_points_x, stop_points_y = [], []
		go_points_x, go_points_y = [], []
		for i in range(1, len(paths[next_path])):
			path = paths[next_path]
			LAT1 = path.LAT.iloc[i-1]
			LAT2 = path.LAT.iloc[i]
			LON1 = path.LON.iloc[i-1]
			LON2 = path.LON.iloc[i]

			dTIME = path.BaseDateTime.iloc[i] - path.BaseDateTime.iloc[i-1]
			'''
			dLAT = (path.LAT.iloc[i] * (LAT_MAX-LAT_MIN) + LAT_MIN) - (path.LAT.iloc[i-1] * (LAT_MAX-LAT_MIN) + LAT_MIN)
			dLON = (path.LON.iloc[i] * (LON_MAX-LON_MIN) + LON_MIN) - (path.LON.iloc[i-1] * (LON_MAX-LON_MIN) + LON_MIN) 
			dTIME = path.BaseDateTime.iloc[i] - path.BaseDateTime.iloc[i-1]
			dPOS = math.sqrt(dLAT*dLAT + dLON*dLON)
			dist_over_earth = distance_over_earth(path.LAT.iloc[i-1], path.LON.iloc[i-1], path.LAT.iloc[i], path.LON.iloc[i])
			point_color = "g"
			point_label = "go"
			if dPOS != 0:
				point_color = "g"
				point_label = "go"
				go_points_x.append(path.LON.iloc[i] * (LON_MAX-LON_MIN) + LON_MIN) 
				go_points_y.append(path.LAT.iloc[i] * (LAT_MAX-LAT_MIN) + LAT_MIN)
			if dPOS == 0:
				point_color = "r"
				point_label = "stop"
				stop_points_x.append(path.LON.iloc[i] * (LON_MAX-LON_MIN) + LON_MIN) 
				stop_points_y.append(path.LAT.iloc[i] * (LAT_MAX-LAT_MIN) + LAT_MIN)		
		#print("stop points: " + str(len(stop_points_x)))
		#print("go points: " + str(len(go_points_x)))
		plt.axis("off")
		#plt.title(str(next_path))
		plt.scatter(go_points_x, go_points_y, label = "go", c="g", s=0.75)
		plt.scatter(stop_points_x, stop_points_y, label = "stop", c="r", s=0.75)
		print("LAT1: " + str(path.LAT.iloc[i-1] * (LAT_MAX-LAT_MIN) + LAT_MIN) \
			+ " - LON1: " + str(path.LON.iloc[i-1] * (LON_MAX-LON_MIN) + LON_MIN) \
			+ " - LAT2: " +str(path.LAT.iloc[i] * (LAT_MAX-LAT_MIN) + LAT_MIN) \
			+ " - LON2: " + str(path.LON.iloc[i] * (LON_MAX-LON_MIN) + LON_MIN))
		#print(str(dPOS)) 
		'''
		distance = haversine((LAT1, LON1), (LAT2, LON2), unit=Unit.NAUTICAL_MILES)
		print(str(distance) + "nm, displacement over " + str(dTIME) + " seconds, translating into " + str(distance / (dTIME/3600)) + " knots average")
		print("Reported speed: " + str(path.SOG.iloc[i] * SPEED_MAX))
		#print("Calculated movement of " + str(dist_over_earth / dTIME ) + " knots per hour")

def plot_stop_go_with_time(paths, start_index):
	# this was originally based on some geod-related math and matrix manipulation
	# in the interest of time and compatibility with DataFrames, I am ignoring this for now
	#print("Call made to unfinished method remove_unusual_tracks")
	#print("length of input path: " + str(len(path)))
	fig = plt.figure()
	#ax = fig.add_subplot(224, projection='3d')

	path_iterator = iter(paths)
	for i in range(start_index):
		next_path = next(path_iterator)
	for i in range(1, 5):
		next_path = next(path_iterator)
		stop_points_x, stop_points_y, stop_points_t = [], [], []
		go_points_x, go_points_y, go_points_t = [], [], []
		ax = fig.add_subplot(2, 2, i, projection="3d")
		path = paths[next_path]
		for i in range(1, len(path)):
			
			LAT1 = path.LAT.iloc[i-1]
			LAT2 = path.LAT.iloc[i]
			LON1 = path.LON.iloc[i-1]
			LON2 = path.LON.iloc[i]

			dLAT = path.LAT.iloc[i] - path.LAT.iloc[i-1]
			dLON = path.LON.iloc[i] - path.LON.iloc[i-1]
			dTIME = path.BaseDateTime.iloc[i] - path.BaseDateTime.iloc[i-1]
			#dPOS = math.sqrt(dLAT*dLAT + dLON*dLON)
			dPOS = haversine((LAT1, LON1), (LAT2, LON2))
			point_color = "g"
			point_label = "go"
			if dPOS != 0:
				point_color = "g"
				point_label = "go"
				go_points_x.append(path.LON.iloc[i] * (LON_MAX-LON_MIN) + LON_MIN) 
				go_points_y.append(path.LAT.iloc[i] * (LAT_MAX-LAT_MIN) + LAT_MIN)
				go_points_t.append(path.BaseDateTime.iloc[i])
			else:
				point_color = "r"
				point_label = "stop"
				stop_points_x.append(path.LON.iloc[i] * (LON_MAX-LON_MIN) + LON_MIN) 
				stop_points_y.append(path.LAT.iloc[i] * (LAT_MAX-LAT_MIN) + LAT_MIN)
				stop_points_t.append(path.BaseDateTime.iloc[i])		
		print("stop points: " + str(len(stop_points_x)) + ", " + str(len(stop_points_y)) + ", " + str(len(stop_points_t)))
		print("go points: " + str(len(go_points_x)) + ", " + str(len(go_points_y)) + ", " + str(len(go_points_t)))
		#plt.axis("off")
		plt.title(str(next_path) + ": " + resolve_vessel_type(paths[next_path].VesselType[0]))
		ax.scatter(go_points_x, go_points_y, go_points_t, c="g", s = 0.5) #, s=0.75)
		ax.scatter(stop_points_x, stop_points_y, stop_points_t, c="r") #, s=0.75)
		#print(str(dPOS) + " displacement over " + str(dTIME) + " seconds")

### Cut discontinuous tracks into smaller tracks
#	If the time between two tracks is greater than a given interval, split the path into two resultant paths
#	This is useful when a ship leaves the map, or simply hops around because that's not what ships do
def cut_discontinuous_tracks(paths, num_hours=2, visualize_cuts=False):
	# this is intended to separate voyages that leave the area of observation
	print("Call made to unfinished method cut_discontinuous_tracks")
	resultant_paths = {}
	paths_to_pop = []

	if visualize_cuts == True:
		fig = plt.figure()
	INTERVAL_MAX = num_hours * 3600 # this converts the time to number of seconds, ugly but whatever
	for path in paths:
		path_num = path
		times = paths[path].BaseDateTime
		duration = times.values[-1] - times.values[0] #paths[path].BaseDateTime.iloc[-1] - paths[path].BaseDateTime.iloc[0]
		# If the path is not long enough to be split, ignore it
		if (duration < INTERVAL_MAX):
			continue
		#print("MMSI: " + str(path) + " Path duration: " + str(duration))

		if (visualize_cuts):
			ax = fig.add_subplot(1, 2, 1)
			im = plt.imread(current_region + ".png")
			ax.imshow(im, extent=[bounds[2],bounds[3], bounds[0], bounds[1]])
			ax.plot((paths[path].LON * (LON_MAX-LON_MIN) + LON_MIN), (paths[path].LAT * (LAT_MAX-LAT_MIN) + LAT_MIN), label=path)
			ax.legend()

		# Get the indeces (-1) where an interval of more than num_hours hours has passed
		intervals = times.values[1:] - times.values[:-1]
		overlong_intervals = np.where(intervals > INTERVAL_MAX)[0]
		overlong_intervals = np.insert(overlong_intervals, 0, 0) # Account for starting point not being included
		overlong_intervals = np.insert(overlong_intervals, len(overlong_intervals), len(paths[path])-1)

		# TODO: Add something to account for the final point in a path and not drop off the end
		#		If there are no splits, show the original path in the post-split figure
		#		Figure out why complete paths are deleted

		print("Intervals: " + str(overlong_intervals))

		for i in range(1, len(overlong_intervals)):
			# update MMSI for new path names, new array IDs for sub-paths resulting from splitting
			path_num = round(path_num+0.1, 3)
			#time_diff = times.values[overlong_intervals[i]+1] - times.values[overlong_intervals[i]-1]
			time_diff = times.values[overlong_intervals[i]] - times.values[overlong_intervals[i]-1]
			print("Interval of ~" + str(int(time_diff/3600)) + " hours found in MMSI: " + str(path) + " - splitting!")
			resultant_paths.update({path_num : paths[path][overlong_intervals[i-1]+1 : overlong_intervals[i]]})

		# Add on the last part of the initial path, which is not accounted for in overlong_intervals
		# If (total length of path) - (the last point in the intervals) is long enough to be considered, add it on
		if overlong_intervals.size > 0:
			if len(paths[path]) - overlong_intervals[len(overlong_intervals)-1] > 20:
				path_num = round(path_num+0.1, 3)
				# Add on everything after the last splitpoint
				resultant_paths.update({path_num : paths[path][overlong_intervals[len(overlong_intervals)-1] : ]}) 

		#	If overlong_intervals contains more than just the 0 point, start destroying thngs
		if (overlong_intervals.size > 1):
			paths_to_pop.append(path)

		# Remove paths that are too short, no sense in keeping them
		rpaths_to_pop = []
		for check_short in resultant_paths:
			if len(resultant_paths[check_short]) < 20:
				rpaths_to_pop.append(check_short)

		print("Number of sub-paths: " + str(len(resultant_paths)) + "\nShort paths to remove: " + str(len(rpaths_to_pop)) + "\nTotal paths: " + str(len(resultant_paths)-len(rpaths_to_pop)))

		for pop in rpaths_to_pop:
			resultant_paths.pop(pop)

		print("Total paths (actual): " + str(len(resultant_paths)))

	if (visualize_cuts):
		ax = fig.add_subplot(1, 2, 2)
		im = plt.imread(current_region + ".png")
		ax.imshow(im, extent=[bounds[2],bounds[3], bounds[0], bounds[1]])
		for rpath in resultant_paths:
			ax.plot((resultant_paths[rpath].LON * (LON_MAX-LON_MIN) + LON_MIN), (resultant_paths[rpath].LAT * (LAT_MAX-LAT_MIN) + LAT_MIN), label=rpath)
		ax.legend()
		plt.show()
	else:
		for victim in paths_to_pop:
			paths.pop(victim)
		#for new_path in resultant_paths:
		paths.update(resultant_paths)
		#return resultant_paths
'''
iterator = iter(paths)
for i in range(40):
	q = next(iterator)
	paths_to_check = {q : paths[q]}
	utils.cut_discontinuous_tracks(paths_to_check, 2, True)
'''

