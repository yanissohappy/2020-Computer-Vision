import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np

img = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
length = len(img) # row
width = len(img[0]) # col, it's actually a square though
	
def binarize(image):
	ret = np.zeros(image.shape, np.int)
	for i in range(int(width)):
		for j in range(int(length)):
			if image[j][i] >= 128: # 128 ~ 255
				ret[j][i] = 255
			else: # 0 ~ 127
				ret[j][i] = 0
	return ret

def downSample(image):
	ret = np.zeros((64, 64), np.int)
	for i in range(0, 64):
		for j in range(0, 64):
			ret[i, j] = image[i * 8, j * 8]
	return ret
	
def h_function(b, c, d, e):
	if b == c and (d != b or e != b):
		return "q"
	if b == c and (d == b and e == b):
		return "r"
	if b != c and (d == b and e == b):
		return "s"
	
def f_function(a1, a2, a3, a4):
	if a1 == a2 == a3 == a4 == "r":
		return 5
	else:
		count = 0
		if a1 == "q": count += 1
		if a2 == "q": count += 1
		if a3 == "q": count += 1
		if a4 == "q": count += 1
		return count
	
def yokoiConnect(down_sample_image):
	ret = np.zeros((64, 64), np.int)
	"""
	7 2 6
	3 0 1
	8 4 5
	"""
	for i in range(64):
		for j in range(64):	
			x = np.zeros((9), np.int)
			x[0] = down_sample_image[i, j]
			if i == 0 and j == 0: # left-upmost
				x[1] = down_sample_image[i, j + 1]
				x[4] = down_sample_image[i + 1, j]
				x[5] = down_sample_image[i + 1, j + 1]
			elif i == 0 and j == 63: # right-upmost
				x[3] = down_sample_image[i, j - 1]
				x[8] = down_sample_image[i + 1, j - 1]
				x[4] = down_sample_image[i + 1, j]
			elif i == 63 and j == 0: # left-bottommost
				x[2] = down_sample_image[i - 1, j]
				x[6] = down_sample_image[i - 1, j + 1]
				x[1] = down_sample_image[i, j + 1]
			elif i == 63 and j == 63: # right-bottommost
				x[2] = down_sample_image[i - 1, j]
				x[7] = down_sample_image[i - 1, j - 1]
				x[3] = down_sample_image[i, j - 1]
			elif i == 0: # upmost
				x[3] = down_sample_image[i, j - 1]
				x[8] = down_sample_image[i + 1, j - 1]
				x[4] = down_sample_image[i + 1, j]
				x[5] = down_sample_image[i + 1, j + 1]
				x[1] = down_sample_image[i, j + 1]
			elif j == 0: # leftmost
				x[2] = down_sample_image[i - 1, j]
				x[6] = down_sample_image[i - 1, j + 1]
				x[1] = down_sample_image[i, j + 1]
				x[5] = down_sample_image[i + 1, j + 1]
				x[4] = down_sample_image[i + 1, j]
			elif i == 63: # bottommost
				x[2] = down_sample_image[i - 1, j]
				x[7] = down_sample_image[i - 1, j - 1]
				x[3] = down_sample_image[i, j - 1]
				x[6] = down_sample_image[i - 1, j + 1]
				x[1] = down_sample_image[i, j + 1]
			elif j == 63: # rightmmost
				x[2] = down_sample_image[i - 1, j]
				x[7] = down_sample_image[i - 1, j - 1]
				x[3] = down_sample_image[i, j - 1]
				x[8] = down_sample_image[i + 1, j - 1]
				x[4] = down_sample_image[i + 1, j]
			else:
				x[2] = down_sample_image[i - 1, j]
				x[7] = down_sample_image[i - 1, j - 1]
				x[3] = down_sample_image[i, j - 1]
				x[6] = down_sample_image[i - 1, j + 1]
				x[1] = down_sample_image[i, j + 1]
				x[4] = down_sample_image[i + 1, j]
				x[5] = down_sample_image[i + 1, j + 1]
				x[8] = down_sample_image[i + 1, j - 1]
			
			if down_sample_image[i, j] == 0:
				ret[i, j] = 0
			else:
				ret[i, j] = f_function( \
							h_function(x[0], x[1], x[6], x[2]), \
                            h_function(x[0], x[2], x[7], x[3]), \
                            h_function(x[0], x[3], x[8], x[4]), \
                            h_function(x[0], x[4], x[5], x[1]))
	return ret

def pairRelationshipOperator(down_sample_image, yokoi_image):
	ret = np.zeros((64, 64), np.int)
	"""
	7 2 6
	3 0 1
	8 4 5
	"""
	for i in range(64):
		for j in range(64):	
			x = np.zeros((5), np.int)
			if i == 0 and j == 0: # left-upmost
				x[1] = yokoi_image[i, j + 1]
				x[4] = yokoi_image[i + 1, j]
			elif i == 0 and j == 63: # right-upmost
				x[3] = yokoi_image[i, j - 1]
				x[4] = yokoi_image[i + 1, j]
			elif i == 63 and j == 0: # left-bottommost
				x[2] = yokoi_image[i - 1, j]
				x[1] = yokoi_image[i, j + 1]
			elif i == 63 and j == 63: # right-bottommost
				x[2] = yokoi_image[i - 1, j]
				x[3] = yokoi_image[i, j - 1]
			elif i == 0: # upmost
				x[3] = yokoi_image[i, j - 1]
				x[4] = yokoi_image[i + 1, j]
				x[1] = yokoi_image[i, j + 1]
			elif j == 0: # leftmost
				x[2] = yokoi_image[i - 1, j]
				x[1] = yokoi_image[i, j + 1]
				x[4] = yokoi_image[i + 1, j]
			elif i == 63: # bottommost
				x[2] = yokoi_image[i - 1, j]
				x[3] = yokoi_image[i, j - 1]
				x[1] = yokoi_image[i, j + 1]
			elif j == 63: # rightmmost
				x[2] = yokoi_image[i - 1, j]
				x[3] = yokoi_image[i, j - 1]
				x[4] = yokoi_image[i + 1, j]
			else:
				x[2] = yokoi_image[i - 1, j]
				x[3] = yokoi_image[i, j - 1]
				x[1] = yokoi_image[i, j + 1]
				x[4] = yokoi_image[i + 1, j]
			
			if down_sample_image[i, j] == 0: # if original image is 0, then output 0
				ret[i, j] = 0
			elif yokoi_image[i, j] != 1: # not edge
				ret[i, j] = 2
			else:
				if np.count_nonzero(x == 1) >= 1: # count how many 1 in x
					ret[i, j] = 1 # p
				else:
					ret[i, j] = 2 # q
	return ret	
	
def connectedShrinkOperator(down_sample_image, pair_image): # form is really similar to Yokoi algo
	"""
	7 2 6
	3 0 1
	8 4 5
	"""
	is_changed = False
	for i in range(64):
		for j in range(64):	
			if pair_image[i, j] == 1 and down_sample_image[i, j] == 255:
				x = np.zeros((9), np.int)
				x[0] = down_sample_image[i, j]
				if i == 0 and j == 0: # left-upmost
					x[1] = down_sample_image[i, j + 1]
					x[4] = down_sample_image[i + 1, j]
					x[5] = down_sample_image[i + 1, j + 1]
				elif i == 0 and j == 63: # right-upmost
					x[3] = down_sample_image[i, j - 1]
					x[8] = down_sample_image[i + 1, j - 1]
					x[4] = down_sample_image[i + 1, j]
				elif i == 63 and j == 0: # left-bottommost
					x[2] = down_sample_image[i - 1, j]
					x[6] = down_sample_image[i - 1, j + 1]
					x[1] = down_sample_image[i, j + 1]
				elif i == 63 and j == 63: # right-bottommost
					x[2] = down_sample_image[i - 1, j]
					x[7] = down_sample_image[i - 1, j - 1]
					x[3] = down_sample_image[i, j - 1]
				elif i == 0: # upmost
					x[3] = down_sample_image[i, j - 1]
					x[8] = down_sample_image[i + 1, j - 1]
					x[4] = down_sample_image[i + 1, j]
					x[5] = down_sample_image[i + 1, j + 1]
					x[1] = down_sample_image[i, j + 1]
				elif j == 0: # leftmost
					x[2] = down_sample_image[i - 1, j]
					x[6] = down_sample_image[i - 1, j + 1]
					x[1] = down_sample_image[i, j + 1]
					x[5] = down_sample_image[i + 1, j + 1]
					x[4] = down_sample_image[i + 1, j]
				elif i == 63: # bottommost
					x[2] = down_sample_image[i - 1, j]
					x[7] = down_sample_image[i - 1, j - 1]
					x[3] = down_sample_image[i, j - 1]
					x[6] = down_sample_image[i - 1, j + 1]
					x[1] = down_sample_image[i, j + 1]
				elif j == 63: # rightmmost
					x[2] = down_sample_image[i - 1, j]
					x[7] = down_sample_image[i - 1, j - 1]
					x[3] = down_sample_image[i, j - 1]
					x[8] = down_sample_image[i + 1, j - 1]
					x[4] = down_sample_image[i + 1, j]
				else:
					x[2] = down_sample_image[i - 1, j]
					x[7] = down_sample_image[i - 1, j - 1]
					x[3] = down_sample_image[i, j - 1]
					x[6] = down_sample_image[i - 1, j + 1]
					x[1] = down_sample_image[i, j + 1]
					x[4] = down_sample_image[i + 1, j]
					x[5] = down_sample_image[i + 1, j + 1]
					x[8] = down_sample_image[i + 1, j - 1]
				
				result = pairRelationship_f( \
								pairRelationship_h(x[0], x[1], x[6], x[2]), \
								pairRelationship_h(x[0], x[2], x[7], x[3]), \
								pairRelationship_h(x[0], x[3], x[8], x[4]), \
								pairRelationship_h(x[0], x[4], x[5], x[1]))
				if result == 1: # can be background and record whether changed
					down_sample_image[i, j] = 0
					is_changed = True
	return down_sample_image, is_changed
	
def pairRelationship_h(b, c, d, e):
	if b == c and (d != b or e != b):
		return 1
	else:
		return 0
		
def pairRelationship_f(a1, a2, a3, a4):
	count = 0
	if a1 == 1: count += 1
	if a2 == 1: count += 1
	if a3 == 1: count += 1
	if a4 == 1: count += 1
	return count	
			
# get binary photo
binarized_photo = binarize(img)
# down sample
downSample_photo = downSample(binarized_photo)

while True:
	is_changed = False
	# yokoi
	yokoiConnect_photo = yokoiConnect(downSample_photo)
	# pair relationship operator
	pairRelationshipOperator_photo = pairRelationshipOperator(downSample_photo, yokoiConnect_photo) # include "p" and "q"
	# connected shrink operator
	downSample_photo, is_changed = connectedShrinkOperator(downSample_photo, pairRelationshipOperator_photo)
	if is_changed == False:
		break

cv2.imwrite('thinning_photo.png', downSample_photo)