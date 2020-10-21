import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np

img = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
length = len(img) # row
width = len(img[0]) # col, it's actually a square though
kernel = np.ones((5, 5), dtype = np.uint8)

def initializeKernel(kernel):
	kernel[0, 0] = 0
	kernel[0, 4] = 0
	kernel[4, 0] = 0
	kernel[4, 4] = 0
	return kernel
	
def binarize(image):
	ret = np.zeros(image.shape, np.int)
	for i in range(int(width)):
		for j in range(int(length)):
			if image[j][i] >= 128: # 128 ~ 255
				ret[j][i] = 255
			else: # 0 ~ 127
				ret[j][i] = 0
	return ret

def dilation(image, kernel):
	ret = np.zeros(image.shape, np.int)
	# center of kernel will be put from [[2, width - 1 - 2], [2, length - 1 - 2]] becuz kernel center is in (2, 2)
	# hence, I can imagine kernel (0, 0) will be put in the (-2, -2) of the original photo (Lena) 
	for i in range(int(width)):
		for j in range(int(length)):
			value = 0
			for kernel_i in range(5):
				for kernel_j in range(5):
					scope_i = i + kernel_i - 2
					scope_j = j + kernel_j - 2
					if kernel[kernel_i, kernel_j] == 1: # if some position is 1 in structure component, then do the following
						if (0 <= scope_i and scope_i < length) and (0 <= scope_j and scope_j < width):
							if image[scope_i][scope_j] == 255:
								value = 255 # make it white
			ret[i, j] = value
	return ret

def erosion(image, kernel):
	ret = np.zeros(image.shape, np.int)
	# center of kernel will be put from [[2, width - 1 - 2], [2, length - 1 - 2]] becuz kernel center is in (2, 2)
	# hence, I can imagine kernel (0, 0) will be put in the (-2, -2) of the original photo (Lena) 
	for i in range(int(width)):
		for j in range(int(length)):
			value = 255
			for kernel_i in range(5):
				for kernel_j in range(5):
					scope_i = i + kernel_i - 2
					scope_j = j + kernel_j - 2
					if kernel[kernel_i, kernel_j] == 1: # if some position is 1 in structure component, then do the following
						if (0 <= scope_i and scope_i < length) and (0 <= scope_j and scope_j < width):
							if image[scope_i][scope_j] == 0: # if one of kernel touches black part
								value = 0 # make it black
			ret[i, j] = value
	return ret
	
def opening(eroted_photo, kernel):
	return dilation(eroted_photo, kernel)

def closing(dilated_photo, kernel):
	return erosion(dilated_photo, kernel)	
	
def complement(image):
	ret = np.zeros(image.shape, np.int)
	for i in range(int(width)):
		for j in range(int(length)):
			if image[i, j] == 255:
				ret[i, j] = 0
			else:
				ret[i, j] = 255
	return ret
	
def intersection(image1, image2):
	ret = np.zeros((length, width), np.int)
	for i in range(int(width)):
		for j in range(int(length)):
			if image1[i, j] == 255 and image2[i, j] == 255:
				ret[i, j] = 255
	return ret
				
	
def hitAndMiss(image): # this algorithm just uses upside-down L structure element
	A_J = np.zeros(image.shape, np.int)
	Ac_K = np.zeros(image.shape, np.int)
	complement_image = complement(image)
	for i in range(int(width)):
		for j in range(int(length)): 
			if (i + 1 < length) and (j - 1 >= 0): # erosion (A (-) J) 
				if image[i, j - 1] == 255 and image[i, j] == 255 and image[i + 1, j] == 255:
					A_J[i, j] = 255
	
			if (i - 1 >= 0) and (j + 1 < width): # erosion (Ac (-) K) 
				if complement_image[i - 1, j] == 255 and complement_image[i - 1, j + 1] == 255 and complement_image[i, j + 1] == 255:
					Ac_K[i, j] = 255
	return intersection(A_J, Ac_K)					
	
	
''' test another way for dilation
def try_d(image, kernel):
	ret = np.zeros(image.shape, np.int)
	# center of kernel will be put from [[2, width - 1 - 2], [2, length - 1 - 2]] becuz kernel center is in (2, 2)
	# hence, I can imagine kernel (0, 0) will be put in the (-2, -2) of the original photo (Lena) 
	for i in range(int(width)):
		for j in range(int(length)):
			if image[i, j] == 255:
				for kernel_i in range(-2, 3, 1):
					for kernel_j in range(-2, 3, 1):
						if (0 <= i + kernel_i and i + kernel_i < length) and (0 <= j + kernel_j and j + kernel_j < width):	
							ret[i + kernel_i, j + kernel_j] = 255
	return ret				
'''

# set kernel	
kernel = initializeKernel(kernel)
# get binary photo
binarized_photo = binarize(img)
# dilation
dilated_photo = dilation(binarized_photo, kernel)
cv2.imwrite('dilation.png', dilated_photo)
# erosion
eroted_photo = erosion(binarized_photo, kernel)
cv2.imwrite('erosion.png', eroted_photo)
# opening
opened_photo = opening(eroted_photo, kernel)
cv2.imwrite('opening.png', opened_photo)
# closing
closed_photo = closing(dilated_photo, kernel)
cv2.imwrite('closing.png', closed_photo)
# hit and miss
hitandmiss_photo = hitAndMiss(binarized_photo)
cv2.imwrite('hit_and_miss.png', hitandmiss_photo)
