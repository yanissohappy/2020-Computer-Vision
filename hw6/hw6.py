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

def printOut(yokoi_connect_picture):
	with open("output.txt", 'w') as out_file:
		for i in range(64):
			for j in range(64):	
				if yokoi_connect_picture[i, j] != 0:
					out_file.write(str(yokoi_connect_picture[i, j]))
				else:
					out_file.write(" ")
				# out_file.write(" ")
			out_file.write("\n")		
							
# get binary photo
binarized_photo = binarize(img)
# down sample
downSample_photo = downSample(binarized_photo)
# yokoi
yokoiConnect_photo = yokoiConnect(downSample_photo)
printOut(yokoiConnect_photo)

