
### Vessel Codes
fishing = [30, 1001, 1002]
passenger = [60, 61, 62, 63, 64, 69]
cargo = [70, 71, 72, 73, 74, 79, 1003,1004,1016]
tanker = [80, 81, 82, 83, 84, 89, 1024]
pleasurecraft = [36, 37, 1019]
speedy = [40, 41, 42, 43, 44, 49]
cargo = [70, 71, 72, 73, 74, 79, 1003, 1004, 1016]
tug = [31, 32, 52, 1025]

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