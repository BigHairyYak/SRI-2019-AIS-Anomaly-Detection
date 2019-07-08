
### Vessel Codes
fishing = [30, 1001, 1002]
passenger = [60, 61, 62, 63, 64, 69]
cargo = [70, 71, 72, 73, 74, 79, 1003,1004,1016]
tanker = [80, 81, 82, 83, 84, 89, 1024]
pleasurecraft = [36, 37, 1019]
speedy = [40, 41, 42, 43, 44, 49]
cargo = [70, 71, 72, 73, 74, 79, 1003, 1004, 1016]
tug = [31, 32, 52, 1025]

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
		return("Unknown") # if nothing is found, again return failure

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