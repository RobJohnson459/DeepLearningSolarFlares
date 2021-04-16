magDict = {
    'TOTUSJH': 0,
    'TOTBSQ': 1,
    'TOTPOT': 2,
    'TOTUSJZ': 3,
    'ABSNJZH': 4,
    'SAVNCPP': 5,
    'USFLUX': 6,
    'TOTFZ': 7,
    'MEANPOT': 8,
    'EPSZ': 9,
    'SHRGT45': 10,
    'MEANSHR': 11,
    'MEANGAM': 12,
    'MEANGBT': 13,
    'MEANGBZ': 14,
    'MEANGBH': 15,
    'MEANJZH': 16,
    'TOTFY': 17,
    'MEANJZD': 18,
    'MEANALP': 19,
    'TOTFX': 20,
    'EPSY': 21,
    'EPSX': 22,
    'R_VALUE': 23,
    'RBZ_VALUE': 24,
    'RBT_VALUE': 25,
    'RBP_VALUE': 26,
    'FDIM': 27,
    'BZ_FDIM': 28,
    'BT_FDIM': 29,
    'BP_FDIM': 30,
    'PIL_LEN': 31,
    'XR_MAX': 32
}

def counter(filename, lines=-1):
	file = open(filename, mode='r')
	# This is an iterator to exit early
	row = 0
	# This dictionary corresponds to the five levels of flare events, from highest to lowest, with Q meaning quiet, or no flare.
	dict = {' "X"':0, ' "M"':1, ' "C"':2, ' "B"':3, ' "Q"':4}


	hist = [0,0,0,0,0]
	for line in file:
		if 'nan' in line or 'NaN' in line:
			continue
		cat = line.split(":")[2].split(',')[0]
		if row == lines:
			return hist
		else:
			hist[dict[cat]] += 1

	file.close()
	return hist



	# Get the data from the JSON file, then return it as a tensor of input data and a list of labels
def getDataFromJSON(path="data/train_partition1_data.json", device='cpu', earlyStop=-1):
	''' 
	path is the path to the files, device is where to store it (CUDA), earlyStop is how many lines to 
	read if you don't want the entire file read. The default is -1, which will read the entire file.
	This function returns a tensor containing the data, a list containing the corresponding labels,
	and a list that contains the counts of each type of solar flare.
	'''
		# This dataset is heavily skewed, so we need to get the number of each type of flare.
	# This also lets us get the number of lines in the file with a sum.
	# This function also ignores any lines with a value of NaN.
	weights = counter(path, earlyStop)
	lines = np.sum(weights)
	# Check when we want to stop - the end of the file or earlier.
	if earlyStop < 0: length = lines
	else: length = min(earlyStop, lines)
	
	# Get the file and open it. 
	file = open(path)	
	
	# Declare a tensor to hold the data, and a list to hold the labels.
	# Dimensions: 0: number of entries we want. 1: the 33 fields in the data. 2: the 60 observations in each field. 
	tnsr = torch.Tensor().new_empty((length, 33, 60), device=device)
	labels = []
	flares = {'X':0, 'M':1, 'C':2, 'B':3, 'Q':4}
		
	row = -1
	for line in file:
		if 'nan' in line or "NaN" in line:
			continue
		# Load the line as a dictionary. Row is an integer place and v is a smaller dictionary.
		d: dict = json.loads(line)
		row += 1
		for _, v in d.items(): # we use the _ because we don't want the ID.
			if earlyStop > 0 and row >= earlyStop:
				# If we don't want the entire dataset, stop loading more than we want
				return tnsr, labels, weights
			if row % 100 == 0:
				print(f'Now loading event {row}/{length}')
			# append the label to our list
			labels.append(flares[v['label']])
			
			# Break each individual dictionary into dictionaries of observations
			# Key is the string in magDict, and timeDict is a dictionary of observations over time
			for key, timeDict in v['values'].items():
				# Turn our name string into a numeric value
				location = magDict[key]
				# Get the measurements out of the time series dictionary
				for timeStamp, measurement in timeDict.items():
					tnsr[row][location][int(timeStamp)] = measurement
	print(f'{row} lines loaded.')
	# Close the file. I'm not a heathen					
	file.close()
	# This might be a good place to perform some post processing, but that's a question for another day.
	# Famous last words.
	return tnsr, labels, weights

