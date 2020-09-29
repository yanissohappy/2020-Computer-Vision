import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np

img = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
length = len(img) # row
width = len(img[0]) # col, it's actually a square though
label = 0

def binarize(image):
	# ret = image.copy() # can't use .copy or bounding box will be wrong
	ret = np.zeros(image.shape, np.int)
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
	
def makeLabel(image):
	# ret = image.copy()
	ret = np.zeros(image.shape, np.int)
	# 1st pass, scanning and make new label
	for i in range(int(length)):
		for j in range(int(width)):
			if image[i][j] != 0: # 255
				ret[i][j] = setValue(ret, i, j)
	return ret

def firstPass(image):
	for i in range(int(length)):
		for j in range(int(width)):
			if image[i][j] != 0: # 255
				_min = image[i][j]
				if i != 0 and j != 0: # left upper
					if image[i - 1][j - 1] != 0 and image[i - 1][j - 1] < _min:
						_min = image[i - 1][j - 1]
				if i != 0: # upper
					if image[i - 1][j] != 0 and image[i - 1][j] < _min:
						_min = image[i - 1][j]
				if i != 0 and j != int(width) - 1: # right upper
					if image[i - 1][j + 1] != 0 and image[i - 1][j + 1] < _min:
						_min = image[i - 1][j + 1]
				if j != 0: # just left
					if image[i][j - 1] != 0 and image[i][j - 1] < _min:
						_min = image[i][j - 1]
				image[i][j] = _min
				
				if i != 0 and j != 0: # left upper
					if image[i - 1][j - 1] != 0:
						image[i - 1][j - 1] = _min
				if i != 0: # upper
					if image[i - 1][j] != 0:
						image[i - 1][j] = _min
				if i != 0 and j != int(width) - 1: # right upper
					if image[i - 1][j + 1] != 0:
						image[i - 1][j + 1] = _min
				if j != 0: # just left
					if image[i][j - 1] != 0:
						image[i][j - 1] = _min	
					

def secondPass(image):	# check right, right down, down, left down
	# image[i][j + 1], image[i + 1][j + 1], image[i + 1][j], image[i + 1][j - 1]
	for i in range(int(length) - 1, -1, -1):
		for j in range(int(width) - 1, -1, -1):
			if image[i][j] != 0: # 255
				_min = image[i][j]
				if i != length - 1 and j != width - 1: # the rightmost and the last row corner
					if image[i + 1][j + 1] != 0 and image[i + 1][j + 1] < _min:
						_min = image[i + 1][j + 1]
				if i != length - 1: # the last row
					if image[i + 1][j] != 0 and image[i + 1][j] < _min:
						_min = image[i + 1][j]
				if i != length - 1 and j != 0: # the leftmost and the last row corner
					if image[i + 1][j - 1] != 0 and image[i + 1][j - 1] < _min:
						_min = image[i + 1][j - 1]
				if j != width - 1: 
					if image[i][j + 1] != 0 and image[i][j + 1] < _min:
						_min = image[i][j + 1]
				image[i][j] = _min
					
				if i != length - 1 and j != width - 1: # the rightmost and the last row corner
					if image[i + 1][j + 1] != 0:
						image[i + 1][j + 1] = _min
				if i != length - 1: # the last row
					if image[i + 1][j] != 0:
						image[i + 1][j] = _min
				if i != length - 1 and j != 0: # the leftmost and the last row corner
					if image[i + 1][j - 1] != 0:
						image[i + 1][j - 1] = _min
				if j != width - 1: # just left
					if image[i][j + 1] != 0:
						image[i][j + 1] = _min
				

def countPixelandSetThreshold(image, pixel_in_a_class):
	for i in range(int(length)):
		for j in range(int(width)):
			if image[i][j] != 0: 
				pixel_in_a_class[image[i][j]] += 1

def updateBoxBorder(top, bottom, left, right, i, j):
	if i < top:
		top = i
	if i > bottom:
		bottom = i
	if j < left:
		left = j
	if j > right:
		right = j		
	return top, bottom, left, right
				
# binary picture output
binarized_photo = binarize(img)
cv2.imwrite('binary.png', binarized_photo)

# histogram picture output
pixel_count = histogramPixelCount(img)
plt.bar(range(256), pixel_count, width = 1, facecolor = "blue", edgecolor = 'white')
plt.savefig("histogram.png")

# Connected Component
ConnectedComponent = makeLabel(binarized_photo)
back_up = copy.deepcopy(ConnectedComponent)

# reorganize ConnectedComponent
while True:
	print("Connected Component Label Propagating and Converging...")
	firstPass(ConnectedComponent)
	secondPass(ConnectedComponent)
	if False not in (ConnectedComponent == back_up):
		break	
	back_up = copy.deepcopy(ConnectedComponent)
	
# threshold == 500
pixel_in_a_class = [0 for _ in range(np.max(ConnectedComponent) + 1)]
countPixelandSetThreshold(ConnectedComponent, pixel_in_a_class)

for i in range(len(pixel_in_a_class)):
	if pixel_in_a_class[i] >= 500:
		sum_j, sum_k = 0, 0 # for centroid use
		top, bottom, left, right = length, -1, width, -1 # initial value
		for j in range(int(length)):
			for k in range(int(width)):
				if (ConnectedComponent[j][k] == i): # if belongs to that label
					sum_j += j
					sum_k += k
					top, bottom, left, right = updateBoxBorder(top, bottom, left, right, j, k)
		row_centroid = round(sum_j / pixel_in_a_class[i])
		col_centroid = round(sum_k / pixel_in_a_class[i])
		cv2.rectangle(binarized_photo, (left, top), (right, bottom), (127, 0, 0), 2)
		cv2.circle(binarized_photo, (col_centroid, row_centroid), 5, (127, 0, 0), -1) # centroid = x, y = (col, row)
cv2.imwrite('boundingBox.png', binarized_photo)