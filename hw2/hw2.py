import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np

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
	
def makeLabel(image):
	ret = image.copy()
	# 1st pass, scanning and make new label
	for i in range(int(length)):
		for j in range(int(width)):
			if ret[i][j] != 0: # 255
				# check left-upper, upper, right-upper, left (if exists)
				# ret[i][j] = setValue(ret, i, j)
				ret[i][j] = newLabel()
	return ret

def firstPass(image):
	for i in range(int(length)):
		for j in range(int(width)):
			if image[i][j] != 0: # 255
				_min, is_checked = image[i][j], 0
				if i != 0 and j != 0: # left upper
					if image[i - 1][j - 1] != 0 and image[i - 1][j - 1] < _min and is_checked != 1:
						_min = image[i - 1][j - 1]
						is_checked = 1
				if i != 0: # upper
					if image[i - 1][j] != 0 and image[i - 1][j] < _min and is_checked != 1:
						_min = image[i - 1][j]
						is_checked = 1
				if i != 0 and j != int(length) - 1: # right upper
					if image[i - 1][j + 1] != 0 and image[i - 1][j + 1] < _min and is_checked != 1:
						_min = image[i - 1][j + 1]
						is_checked = 1
				if j != 0: # just left
					if image[i][j - 1] != 0 and image[i][j - 1] < _min and is_checked != 1:
						_min = image[i][j - 1]
						is_checked = 1
				if _min != image[i][j]:
					image[i][j] = _min	

def secondPass(image):	# adjust all equivalence_list
	for i in range(int(length) - 1, -1, -1):
		for j in range(int(width) - 1, -1, -1):
			if image[i][j] != 0: # 255
				_min, is_checked = image[i][j], 0
				if i != 0 and j != 0: # left upper
					if image[i - 1][j - 1] != 0 and image[i - 1][j - 1] < _min and is_checked != 1:
						_min = image[i - 1][j - 1]
						is_checked = 1
				if i != 0: # upper
					if image[i - 1][j] != 0 and image[i - 1][j] < _min and is_checked != 1:
						_min = image[i - 1][j]
						is_checked = 1
				if i != 0 and j != int(length) - 1: # right upper
					if image[i - 1][j + 1] != 0 and image[i - 1][j + 1] < _min and is_checked != 1:
						_min = image[i - 1][j + 1]
						is_checked = 1
				if j != 0: # just left
					if image[i][j - 1] != 0 and image[i][j - 1] < _min and is_checked != 1:
						_min = image[i][j - 1]
						is_checked = 1
				if _min != image[i][j]:
					image[i][j] = _min	

def countPixelandSetThreshold(image, pixel_count):
	for i in range(int(length)):
		for j in range(int(width)):
			if image[i][j] != 0: # 255	
				pixel_count[image[i][j]] += 1
	for i in range(int(length)):
		for j in range(int(width)):
			if pixel_count[image[i][j]] < 500: # threshold
				image[i][j] = 0
				
# binary picture output
binarized_list = binarize(img)
cv2.imwrite('binary.jpg', binarized_list)

# histogram picture output
pixel_count = histogramPixelCount(img)
plt.bar(range(256), pixel_count, width = 1, facecolor = "blue", edgecolor = 'white')
plt.savefig("histogram.png")

# Connected Component
ConnectedComponent = makeLabel(binarized_list)
back_up = copy.deepcopy(ConnectedComponent)

# reorganize ConnectedComponent
while True:
	is_loop_again = firstPass(ConnectedComponent)
	secondPass(ConnectedComponent)
	if False not in (ConnectedComponent == back_up):
		break	
	back_up = copy.deepcopy(ConnectedComponent)
	
# threshold == 500
pixel_count = [0 for _ in range(np.max(ConnectedComponent) + 1)]
countPixelandSetThreshold(ConnectedComponent, pixel_count)
for i in range(len(pixel_count)):
	if pixel_count[i] >= 500:
		sum_j, sum_k = 0, 0 # for centroid use
		top, bottom, left, right = length, -1, width, -1 # initial value
		for j in range(int(length)):
			for k in range(int(width)):
				if (ConnectedComponent[j][k] == i):
					sum_j += j
					sum_k += k
					if j < top:
						top = j
					if j > bottom:
						bottom = j
					if k < left:
						left = k
					if k > right:
						right = k	
		row_centroid = round(sum_j / pixel_count[i])
		col_centroid = round(sum_k / pixel_count[i])
		cv2.rectangle(binarized_list, (left, top), (right, bottom), (127, 0, 0), 3)
		cv2.circle(binarized_list, (col_centroid, row_centroid), 4, (127, 0, 0), -1)
cv2.imwrite('boundingBox.jpeg', binarized_list)