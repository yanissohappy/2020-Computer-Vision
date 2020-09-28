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
	if row == 0 and col == 0: # top 1st index
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
	ret = image.copy()
	# ret = [[0 for _ in range(int(width))] for _ in range(int(length))]
	# 1st pass, scanning and make new label
	for i in range(int(length)):
		for j in range(int(width)):
			if ret[i][j] != 0: # 255
				# check left-upper, upper, right-upper, left (if exists)
				# ret[i][j] = setValue(ret, i, j)
				ret[i][j] = newLabel()
	return ret

def firstPass(image, equivalence_list):
	is_revised = 0
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
					equivalence_list[image[i][j]] = _min # change equivalence table
					image[i][j] = _min	
					is_revised = 1
	if is_revised == 1:
		return True
	else:
		return False

def secondPass(image, equivalence_list):	# adjust all equivalence_list
	for i in range(int(length)):
		for j in range(int(width)):
			if image[i][j] != 0: # 255
				temp = equivalence_list[image[i][j]]
				if i != 0 and j != 0: # left upper
					if equivalence_list[image[i - 1][j - 1]] != temp:
						equivalence_list[image[i - 1][j - 1]] = temp
				if i != 0: # upper
					if equivalence_list[image[i - 1][j]] != temp:
						equivalence_list[image[i - 1][j]] = temp
				if i != 0 and j != int(length) - 1: # right upper
					if equivalence_list[image[i - 1][j + 1]] != temp:
						equivalence_list[image[i - 1][j + 1]] = temp
				if j != 0: # just left
					if equivalence_list[image[i][j - 1]] != temp:
						equivalence_list[image[i][j - 1]] = temp

def makeConnectedComponent(image, equivalence_list, pixel_count_for_a_connected):
	for i in range(int(length)):
		for j in range(int(width)):	
			if image[i][j] != 0: # 255
				image[i][j] = equivalence_list[image[i][j]]
				pixel_count_for_a_connected[image[i][j]] += 1
						
# binary picture output
binarized_list = binarize(img)
cv2.imwrite('binary.jpg', binarized_list)

# histogram picture output
pixel_count = histogramPixelCount(img)
plt.bar(range(256), pixel_count, width = 1, facecolor = "blue", edgecolor = 'white')
plt.savefig("histogram.png")

# Connected Component
ConnectedComponent = makeLabel(binarized_list)
print(ConnectedComponent)
equivalence_list = [i for i in range(label + 100)] # used for find the same min value in a pass
while True:
	#print("X")
	is_loop_again = firstPass(ConnectedComponent, equivalence_list)
	secondPass(ConnectedComponent, equivalence_list)
	if is_loop_again == True:
		continue
	else:
		break
pixel_count_for_a_connected = [0 for _ in range(label + 100)] # used for counting number of a connected component
makeConnectedComponent(ConnectedComponent, equivalence_list, pixel_count_for_a_connected)
