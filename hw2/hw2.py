import cv2
import matplotlib.pyplot as plt
import copy

img = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
length = len(img) # row
width = len(img[0]) # col, it's actually a square though
label = 0

def binarize(image):
	ret = image.copy()
	for i in range(int(width)):
		for j in range(int(length)):
			if image[j][i] >= 128: # 128 ~ 255
				ret[j][i] = 255
			else: # 0 ~ 127
				ret[j][i] = 0
	return ret
	
def histogramPixelCount(image):
	ret = [0 for _ in range(256)]
	for i in range(int(width)):
		for j in range(int(length)):
			ret[image[j][i]] += 1 # pixel 0 ~ 255 will be put in corresponding index in "ret" list 
	return ret

def newLabel():
	global label
	label += 1
	return label
	
def setValue(image, row, col):
	if row == 0 and col == 0: # top
		return newLabel()
	if row == 0: # top beside 1st index
		_min = image[row][col - 1]
		if _min != 0:
			return _min
		else:
			return newLabel()
	if row != 0 and col == 0: # left beside 1st index
		temp = [image[row - 1][col], image[row - 1][col + 1]]
		if temp.count(0) == 2:
			return newLabel()
		else: # At least one of them is not 0
			if temp.count(0) == 1:
				temp.remove(0)
				return min(temp)
			elif temp.count(0) == 0:
				return min(temp)
			
	if row != 0 and col == width - 1: # right not first row
		temp = [image[row - 1][col - 1], image[row - 1][col], image[row][col - 1]]
		if temp.count(0) == 3:
			return newLabel()
		else: # At least one of them is not 0
			if temp.count(0) == 2:
				temp.remove(0)
				temp.remove(0)
				return min(temp)
			elif temp.count(0) == 1:
				temp.remove(0)
				return min(temp)
			elif temp.count(0) == 0:
				return min(temp)
				
	# other cases
	temp = [image[row - 1][col - 1], image[row - 1][col], image[row - 1][col + 1], image[row][col - 1]]
	if temp.count(0) == 4:
		return newLabel()
	else: # At least one of them is not 0
		if temp.count(0) == 3:
			temp.remove(0)
			temp.remove(0)
			temp.remove(0)
			return temp[0]
		elif temp.count(0) == 2:
			temp.remove(0)
			temp.remove(0)
			return min(temp)
		elif temp.count(0) == 1:
			temp.remove(0)
			return min(temp)
		elif temp.count(0) == 0:
			return min(temp)
	
def makeConnectedComponent(image):
	ret = image.copy()
	# 1st pass, scanning and make new label
	for i in range(int(length)):
		for j in range(int(width)):
			if ret[i][j] != 0: # 255
				# check left-upper, upper, right-upper, left (if exists)
				ret[i][j] = setValue(ret, i, j)
	return ret

def eightCheck(image, row, col, list_set):
	if row == 0: # top beside 1st index
		if image[row][col - 1] != 0 and image[row][col - 1] != image[row][col]:
			if {image[row][col], image[row][col - 1]} not in list_set:
				list_set.append({image[row][col - 1], image[row][col]})
	elif row != 0 and col == 0: # left beside 1st index
		if image[row - 1][col] != 0 and image[row - 1][col] != image[row][col]:
			if {image[row][col], image[row - 1][col]} not in list_set:
				list_set.append({image[row - 1][col], image[row][col]})
		if image[row - 1][col + 1] != 0 and image[row - 1][col + 1] != image[row][col]:
			if {image[row][col], image[row - 1][col + 1]} not in list_set:
				list_set.append({image[row - 1][col + 1], image[row][col]})
			
	elif row != 0 and col == width - 1: # right not first row
		if image[row - 1][col - 1] != 0 and image[row - 1][col - 1] != image[row][col]:
			if {image[row][col], image[row - 1][col - 1]} not in list_set:
				list_set.append({image[row - 1][col - 1], image[row][col]})
		if image[row - 1][col] != 0 and image[row - 1][col] != image[row][col]:
			if {image[row][col], image[row - 1][col]} not in list_set:
				list_set.append({image[row - 1][col], image[row][col]})
		if image[row][col - 1] != 0 and image[row][col - 1] != image[row][col]:
			if {image[row][col], image[row][col - 1]} not in list_set:
				list_set.append({image[row][col - 1], image[row][col]})
	else:
		# other cases
		if image[row - 1][col - 1] != 0 and image[row - 1][col - 1] != image[row][col]:
			if {image[row][col], image[row - 1][col - 1]} not in list_set:
				list_set.append({image[row - 1][col - 1], image[row][col]})
		if image[row - 1][col] != 0 and image[row - 1][col] != image[row][col]:
			if {image[row][col], image[row - 1][col]} not in list_set:
				list_set.append({image[row - 1][col], image[row][col]})
		if image[row - 1][col + 1] != 0 and image[row - 1][col + 1] != image[row][col]:
			if {image[row][col], image[row - 1][col + 1]} not in list_set:
				list_set.append({image[row - 1][col + 1], image[row][col]})
		if image[row][col - 1] != 0 and image[row][col - 1] != image[row][col]:
			if {image[row][col], image[row][col - 1]} not in list_set:
				list_set.append({image[row][col - 1], image[row][col]})

def mergeConnectPair(list_set):
	pool = set(map(frozenset, list_set))
	groups = []
	while pool:
		groups.append(set(pool.pop()))
		while True:
			for candidate in pool:
				if groups[-1] & candidate:
					groups[-1] |= candidate
					pool.remove(candidate)
					break
			else:
				break	
	return groups
	
def checkConnect(image):
	ret = []
	for i in range(int(length)):
		for j in range(int(width)):
			if image[i][j] != 0: # 255
				eightCheck(image, i, j, ret)
	return ret
	
def updateLabel(image, group):
	for i in range(int(length)):
		for j in range(int(width)):
			if image[i][j] != 0: # 255
				for k in range(len(group)):
					if image[i][j] in group[k]:
						image[i][j] = min(group[k])
	

# binary picture output
binarized_list = binarize(img)
cv2.imwrite('binary.jpg', binarized_list)

# histogram picture output
pixel_count = histogramPixelCount(img)
plt.bar(range(256), pixel_count, width = 1, facecolor = "blue", edgecolor = 'white')
plt.savefig("histogram.png")

# Connected Component
ConnectedComponent = makeConnectedComponent(binarized_list)
group = checkConnect(ConnectedComponent) ####
# print(len(mergeConnectPair(group)[0]))
updateLabel(ConnectedComponent, group)
print(ConnectedComponent[-1])