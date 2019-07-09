import pandas as pd
import matplotlib.pyplot as plt
import mplcursors
from labellines import labelLine, labelLines

### Global Variables for use 
#	Vessel Codes
fishing = [30, 1001, 1002]
passenger = [60, 61, 62, 63, 64, 69]
cargo = [70, 71, 72, 73, 74, 79, 1003,1004,1016]
tanker = [80, 81, 82, 83, 84, 89, 1024]
pleasurecraft = [36, 37, 1019]
speedy = [40, 41, 42, 43, 44, 49]
tug = [31, 32, 52, 1025]

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
	bounds = regions[current_region]
	LAT_MIN = bounds[0]
	LAT_MAX = bounds[1]
	LON_MIN = bounds[2]
	LON_MAX = bounds[3]
	return bounds

# Establish default values of boundaries, if ever used
bounds = get_boundaries()

### Plot vessel courses over a map, to help visualize
#	First establish boundaries to plot between
#	Then show image, then plot the requested number of vessels
def plot_over_map(paths = {}, num_vessels = 20, show_map=True):
	plt.ylim(top=bounds[1], bottom=bounds[0])
	plt.xlim(left=bounds[2], right=bounds[3])
	if show_map:
		im = plt.imread(current_region + ".png")
		plt.imshow(im, extent=[bounds[2],bounds[3], bounds[0], bounds[1]])
	path_iterator = iter(paths)
	for i in range(num_vessels):
		path = next(path_iterator)
		plt.plot(paths[path].LON * (LON_MAX-LON_MIN) + LON_MIN, paths[path].LAT * (LAT_MAX-LAT_MIN) + LAT_MIN, label=path)
		plt.text(paths[path].LON.iloc[-1] * (LON_MAX-LON_MIN) + LON_MIN, paths[path].LAT.iloc[-1] * (LAT_MAX-LAT_MIN) + LAT_MIN, path)

	#mplcursors.cursor(hover=True) # to be able to mouseover
	#labelLines(plt.gca().get_lines(),align=False,fontsize=14)
	plt.show()

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
		return("Pleasure/Sail")
	elif vType in speedy:
		return("High-speed")
	elif vType in passenger:
		return("Passenger")
	elif vType == 1010:
		return("OSV")
	else:
		return("Unknown") # if nothing is found, report as such

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
			plt.title(ais_utils.resolve_vessel_type(values[0, 6])) 
		else:
			plt.title(str(int(values[0, 6])) + ": " + ais_utils.resolve_vessel_type(values[0, 6]))
	plt.show() # Show off the first 25 ship tracks