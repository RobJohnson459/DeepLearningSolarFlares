from datetime import datetime
import random
import torch
import json
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
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

def norm33(X):
	'''
	A simple normalizer that normalizes each data field to be between 0 and 1
	'''
	for i in range(X.shape[1]): # the number of types of measurements
		X[:,i,:] -= torch.min(X[:,i,:])
		X[:,i,:] /= torch.max(X[:,i,:]).item()
	return X

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
def getDataFromJSON(path="data/train_partition1_data.json", earlyStop=-1, device='cpu', test=False):
	''' 
	path is the path to the files, device is where to store it (CUDA), earlyStop is how many lines to 
	read if you don't want the entire file read. The default is -1, which will read the entire file.
	This function returns a tensor containing the data, a list containing the corresponding labels,
	and a list that contains the counts of each type of solar flare.
	
	returns a normalized tensor of inputs, the labels for training/validation data or the ID for test data, and a list of weights.
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
		for id, v in d.items(): 
			if earlyStop > 0 and row >= earlyStop:
				# If we don't want the entire dataset, stop loading more than we want
				return tnsr, labels, weights
			if row % 100 == 0:
				print(f'Now loading event {row}/{length}')
			# append the label to our list
			if test:
				labels.append(id)
			else:
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
	return norm33(tnsr), labels, weights


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
	return norm33(tnsr), labels

def trainer(modelModule, inputs, labels, weight, valSets, valLabels, valweight, *modelArgs, lr=0.0001, epochs=50,
		  pmodel = False, pstateDict = False, pindvLoss = False, pmodelDict = False, pvalidateOut = False,
		  pbl = False, checkClone = False, pbalApp = False, **modelKwargs):
	
	'''
	function call: 
	train(modelModule, inputs, labels, weight, valSets, valLabels, valweight, *modelArgs, lr=0.0001, 
		  pmodel = False, pstateDict = False, pindvLoss = False, pmodelDict = False, pvalidateOut = False,
		  pbl = False, checkClone = False **modelKwargs):
	
	modelModule is the nn.module form of the network you want trained
	inputs and labels are the Xs and Ys respectively of the training data
	weights are the weights to use for the cross entropy loss function - see more in counter documentation
	valinputs, valLabels, and valweight are the same fields but for our validation set
	lr is the learning rate for the optimizer
	Epochs is the amount of times a network is trained on the data
	pmodel prints the model if True, defaults to False
	pstateDict prints the state dict for the optimizer if True, defaults to False
	pindvLoss prints the individual loss for each minibatch if True, defaults to False
	pmodelDict prints the state dictionary for the model if True, defaults to False
	pvalidateOut prints all errors and accuracies of the validation set if True, defaults to False
	pbl prints the batch loss at declaration time if True, defaults to False
	checkClone prints the maximum and the minimum difference between the initial model and the current iteration 
		of the model if True, defaults to False
	pblApp checks the validation appending
	modelArgs and modelKwargs are the arguments for the model and the keyword arguments of the model, repsectivley
	'''

	# Define the model
	model = modelModule(*modelArgs, **modelKwargs)
	# if we are going to check this against itself to make sure it is learning, we need to be able to come back
	if checkClone: 
		PATH = './.cloneModel.pth'
		torch.save(model.state_dict(), PATH)
	if pmodel: print(model)
	
	# Define loss functions
	if weight is not None: weight = torch.Tensor(weight)
	lfc = nn.CrossEntropyLoss(weight=weight)
	if valweight is not None: valweight = torch.Tensor(valweight)
	valLoss = nn.CrossEntropyLoss(weight=valweight)

	

	
	# Hyperparameters
	batch = 64
	
	# Start a dataloader object
	data = list(zip(inputs,labels))
	val = list(zip(valSets,valLabels))
	loader = DataLoader(data, batch_size = batch, num_workers=4)
	valLoader = DataLoader(val, batch_size = 1, num_workers=4)
	opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.25)
	if pstateDict: print(opt.state_dict())

	for epoch in range(epochs):
		model.train()
		batch_loss = []
		if pbl: print(batch_loss)
		for (xtrain,ytrain) in loader:
			opt.zero_grad()
			if pstateDict: print('Opt dict post zero grad:', '================',opt.state_dict(), sep='\n')
			output = model(xtrain)
			loss = lfc(output,ytrain)
			if pindvLoss: print(loss)
			loss.backward()
			if pstateDict: print('Opt dict post backward:', '================',opt.state_dict(), sep='\n')
			if pmodelDict: print(model.state_dict())
			opt.step()
			if pstateDict: print('Opt dict post step:', '================',opt.state_dict(), sep='\n')
			if pstateDict or pmodelDict: print('\n\n\n\n\n')
			batch_loss.append(loss.item())
		print(f'The training loss for epoch {epoch+1}/{epochs} was {np.mean(batch_loss)}')
		if pbl: print(batch_loss)
		
		model.eval()
		balanced = [[],[],[],[],[]]
		batchLoss = []
		unbalanced = []
		
		for (xval,yval) in valLoader:
			output = model(xval)
			loss = valLoss(output,yval)
			batchLoss.append(loss.item())
			guesses = torch.argmax(output,1)
			if pvalidateOut : print('output: \n',output)
			corrects = yval.clone().detach() == guesses
			if pvalidateOut: print(yval.clone().detach(), guesses)
			if pvalidateOut: print(corrects.detach())
			if pvalidateOut: print('===========================\n\n\n')
			unbalanced.append([1 if correct else 0 for correct in corrects.detach()])
		
			for i, ans in enumerate(yval):
				if pbalApp: print(i,ans, guesses[i], corrects[i])
				balanced[ans].append(corrects[i])
		
		balanced = [np.mean(i) for i in balanced]
		balancedAccuracy = np.mean(balanced)
		
		print(f'The total balanced accuracy for validation was {balancedAccuracy}')
		print(f'The validation loss was :   {epoch+1}/{epochs} was {np.mean(batchLoss)}')
		print(f'The unbalanced validation accuracy is {np.mean(unbalanced)}')
		print(f'The accuracy for each is {balanced}')	   
		
		
		if checkClone:
			fives = modelModule(*modelArgs, **modelKwargs)
			fives.load_state_dict(torch.load(PATH))
			s=2
			feature_extraction1 = [child for child in model.children()]
			print(feature_extraction1[s])
			feature_extraction2 = [child for child in fives.children()]
			print(feature_extraction2[s])
			print(torch.max(feature_extraction1[s].weight - feature_extraction2[s].weight).detach())
			print(torch.min(feature_extraction1[s].weight - feature_extraction2[s].weight).detach())
		print('\n\n=============End Epoch==============\n\n')
	if pstateDict: print(opt.state_dict())


	return model

def getTotalAccuracy(m1, m2, m3, x, y): #, unbalanced=True
	o1 = torch.argmax(m1(x))
	o2 = torch.argmax(m2(x))
	o3 = torch.argmax(m3(x))
	c1 = o1 == y.clone().detach()
	c2 = o2 == y.clone().detach()
	c3 = o3 == y.clone().detach()
	return np.mean(np.mean(c3),np.mean(c2),np.mean(c1))

def tester(model, pathToWrite=None):
	if pathToWrite is None:
		pathToWrite = f'results/submission{datetime.now().strftime("%d_%H:%M")}.csv'
	# Get test data
	test, ids, _ = getDataFromJSON(path='data/test_4_5_data.json', test=True, device=device)
	# get our guesses from the network
	guesses = torch.argmax(model(test))
	assert len(ids) == guesses.shape
	# Open a file to write to
	file = open(pathToWrite, mode='w')
	print('Id,Label', file=file)
	for i in range(len(ids)):
		print(ids[i], guesses[i], sep=',', file=file)
	file.close()
