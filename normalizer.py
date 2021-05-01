import random
import torch
import json
import numpy as np
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
flares = {'X':0, 'M':1, 'C':2, 'B':3, 'Q':4}


def counter(filename, lines=-1):
	'''
	This function counts the different types of flares in <filename> to help calculate the weights.
	I'm not totally sure what a good weight calculation is, so here are some ideas

	#ideas
	# 1-(weight/np.sum(weight))
	# .2/weight - this one normalizes so that each class is responsible for 20% of the loss
	# 1/weight - this is a bit naive, but the classes with fewer items are weighted more.
	# 1/(weight+1) - makes sure we don't have any pesky zeroes
	# np.sum(weight)/weight if your learning rate is too low.
	'''
	file = open(filename, mode='r')
	print(f'Now reading {filename}')
	# This is an iterator to exit early
	row = 0
	# This dictionary corresponds to the five levels of flare events, from highest to lowest, with Q meaning quiet, or no flare.


	hist = [0,0,0,0,0]
	for line in file:
		if 'nan' in line or 'NaN' in line:
			continue
		# This line splits the line into keys and such, then splits that for values, then gets the correct character out
		cat = line.split(":")[2].split(',')[0][2]
		if row == lines:
			return hist
		else:
			hist[flares[cat]] += 1
			row += 1

	file.close()
	return hist



	# Get the data from the JSON file, then return it as a tensor of input data and a list of labels
def getDataFromJSON(path="data/train_partition1_data.json", earlyStop=-1, device='cpu'):
	''' 
	path is the path to the files, device is where to store it (CUDA), earlyStop is how many lines to 
	read if you don't want the entire file read. The default is -1, which will read the entire file.
	This function returns a tensor containing the data, a list containing the corresponding labels,
	and a list that contains the counts of each type of solar flare.
	'''
	# This dataset is heavily skewed, so we need to get the number of each type of flare.
	# This also lets us get the number of lines in the file with a sum.
	# This function also ignores any lines with a value of NaN.
	# This function also only gets the amount of lines we either want or need - it will stop at the earlier of earlystop and the end of file.
	weights = counter(path, earlyStop)
	length = np.sum(weights)
	# Check when we want to stop - the end of the file or earlier.
	if earlyStop < 0: length = lines
	else: length = min(earlyStop, lines)
	
	# Get the file and open it. 
	file = open(path)	
	
	# Declare a tensor to hold the data, and a list to hold the labels.
	# Dimensions: 0: number of entries we want. 1: the 33 fields in the data. 2: the 60 observations in each field. 
	tnsr = torch.Tensor().new_empty((length, 33, 60), device=device)
	labels = []
		
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


def subSample(path="data/train_partition1_data.json", earlyStop=-1, device='cuda'):
	# find where to stop - the minimum of our data, or the early stop
	weights = counter(path)
	stop = min(weights)
	if earlyStop > 0: stop = min(stop, earlyStop)

	file = open(path, mode='r') # standard read only file object
	pos = 0 # this will update with the position of a line that we need read
	indicies = [[],[],[],[],[]] # this list will store the locations of lines with an i level flare

	for line in file:
		if 'nan' in line or 'NaN' in line:
			# Lines with NAN  would take more effort to use than it's worth. 
			pos += len(line)
			continue
		# get the category, use the number assigned to it to place the line beginning in the correct list in indicies
		cat = line.split(":")[2].split(',')[0][2] 
		indicies[flares[cat]].append(pos)
		pos += len(line)

	# Choose which lines to read with a random sample, then shuffle them to get a real dataset back
	finalIndicies = []
	for i in range(5):
		finalIndicies += random.sample(indicies[i], stop)
	random.shuffle(finalIndicies)

	tnsr = torch.Tensor().new_empty((stop*5, 33, 60), device=device) #Declare a tensor that can hold all our data
	pos = -1 # reuse this variable, since we don't need it anymore for what it was doing
	labels = [] # list to hold the labels
	# read in the lines dictated by finalIndicies
	for location in finalIndicies:
		file.seek(location)
		line=file.readline()

		# Load the line as a dictionary. Row is an integer place and v is a smaller dictionary.
		d: dict = json.loads(line)
		pos += 1
		if pos % 100 == 0:
			print(f'Now loading event {pos+1}/{stop*5}')
		for _, v in d.items(): # we use the _ because we don't want the ID.
			# append the label to our list
			labels.append(flares[v['label']])
			
			# Break each individual dictionary into dictionaries of observations
			# Key is the string in magDict, and timeDict is a dictionary of observations over time
			for key, timeDict in v['values'].items():
				# Turn our name string into a numeric value
				location = magDict[key]
				# Get the measurements out of the time series dictionary
				for timeStamp, measurement in timeDict.items():
					tnsr[pos][location][int(timeStamp)] = measurement
	print(f'{pos+1} lines loaded.')
	return tnsr, labels
