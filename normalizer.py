

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
