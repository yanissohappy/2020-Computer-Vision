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

def dilation(image, kernel):
	ret = np.zeros(image.shape, np.int)
	# center of kernel will be put from [[2, width - 1 - 2], [2, length - 1 - 2]] becuz kernel center is in (2, 2)
	# hence, I can imagine kernel (0, 0) will be put in the (-2, -2) of the original photo (Lena) 
	for i in range(int(width)):
		for j in range(int(length)):
			_max = []
			for kernel_i in range(5):
				for kernel_j in range(5):
					scope_i = i + kernel_i - 2
					scope_j = j + kernel_j - 2
					if kernel[kernel_i, kernel_j] == 1: # if some position is 1 in structure component, then do the following
						if (0 <= scope_i and scope_i < length) and (0 <= scope_j and scope_j < width):
							_max.append(image[scope_i, scope_j])
			ret[i, j] = max(_max)
	return ret

def erosion(image, kernel):
	ret = np.zeros(image.shape, np.int)
	# center of kernel will be put from [[2, width - 1 - 2], [2, length - 1 - 2]] becuz kernel center is in (2, 2)
	# hence, I can imagine kernel (0, 0) will be put in the (-2, -2) of the original photo (Lena) 
	for i in range(int(width)):
		for j in range(int(length)):
			_min = []
			for kernel_i in range(5):
				for kernel_j in range(5):
					scope_i = i + kernel_i - 2
					scope_j = j + kernel_j - 2
					if kernel[kernel_i, kernel_j] == 1: # if some position is 1 in structure component, then do the following
						if (0 <= scope_i and scope_i < length) and (0 <= scope_j and scope_j < width):
							_min.append(image[scope_i, scope_j])
			ret[i, j] = min(_min)
	return ret
	
def opening(eroted_photo, kernel):
	return dilation(eroted_photo, kernel)

def closing(dilated_photo, kernel):
	return erosion(dilated_photo, kernel)	
	

# set kernel	
kernel = initializeKernel(kernel)
# dilation
dilated_photo = dilation(img, kernel)
cv2.imwrite('dilation.png', dilated_photo)
# erosion
eroted_photo = erosion(img, kernel)
cv2.imwrite('erosion.png', eroted_photo)
# opening
opened_photo = opening(eroted_photo, kernel)
cv2.imwrite('opening.png', opened_photo)
# closing
closed_photo = closing(dilated_photo, kernel)
cv2.imwrite('closing.png', closed_photo)
