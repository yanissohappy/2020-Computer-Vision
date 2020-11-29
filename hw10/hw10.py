import cv2
import matplotlib.pyplot as plt
import random
import numpy as np
import math

img = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
length = len(img) # row
width = len(img[0]) # col, it's actually a square though

def padding(img, symmetric_add): # symmetric_add == 2 if using mask of size 3, symmetric_add == 4 if using mask of size 5
	ret = np.zeros((img.shape[0] + 2, img.shape[1] + 2), np.int)
	# corner
	ret[0, 0] = img[0, 0]
	ret[0, img.shape[1] + 2 - 1] = img[0, img.shape[1] - 1]
	ret[img.shape[0] + 2 - 1, 0] = img[img.shape[0] - 1, 0]
	ret[img.shape[0] + 2 - 1, img.shape[1] + 2 - 1] = img[img.shape[0] - 1, img.shape[1] - 1]
	
	# left and right padding
	for i in range(img.shape[0]):
		ret[i + 1, 0] = img[i , 0]
		ret[i + 1, img.shape[1] + 2 - 1] = img[i, img.shape[1] - 1]
	# upper and down padding
	for j in range(img.shape[1]):
		ret[0, j + 1] = img[0, j]
		ret[img.shape[0] + 2 - 1, j + 1] = img[img.shape[0] - 1, j]
	# interior part	
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			ret[i + 1, j + 1] = img[i, j]
		
	if symmetric_add == 2:
		return ret
	else: 
		symmetric_add -= 2
		return padding(ret, symmetric_add)

def laplace1Operator(padded_img, threshold):
	ret = np.zeros(padded_img.shape, np.int) 
	mask = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
	
	for i in range(padded_img.shape[0] - 2):
		for j in range(padded_img.shape[1] - 2):
			temp = 0
			for m in range(3):
				for n in range(3):
					temp += mask[m, n] * padded_img[i + m, j + n]
			# slide p.165
			if temp >= threshold:
				ret[i, j] = 1
			elif temp <= -threshold:
				ret[i, j] = -1
			else:
				ret[i, j] = 0
	return ret			

def laplace2Operator(padded_img, threshold):
	ret = np.zeros(padded_img.shape, np.int) 
	mask = np.array([[1 / 3, 1 / 3, 1 / 3], [1 / 3, -(8 / 3), 1 / 3], [1 / 3, 1 / 3, 1 / 3]])
	
	for i in range(padded_img.shape[0] - 2):
		for j in range(padded_img.shape[1] - 2):
			temp = 0
			for m in range(3):
				for n in range(3):
					temp += mask[m, n] * padded_img[i + m, j + n]
			# slide p.165
			if temp >= threshold:
				ret[i, j] = 1
			elif temp <= -threshold:
				ret[i, j] = -1
			else:
				ret[i, j] = 0
	return ret			

def minimumVarianceLaplacianOperator(padded_img, threshold):
	ret = np.zeros(padded_img.shape, np.int) 
	mask = np.array([[2 / 3, -(1 / 3), 2 / 3], [-(1 / 3), -(4 / 3), -(1 / 3)], [2 / 3, -(1 / 3), 2 / 3]])
	
	for i in range(padded_img.shape[0] - 2):
		for j in range(padded_img.shape[1] - 2):
			temp = 0
			for m in range(3):
				for n in range(3):
					temp += mask[m, n] * padded_img[i + m, j + n]
			# slide p.165
			if temp >= threshold:
				ret[i, j] = 1
			elif temp <= -threshold:
				ret[i, j] = -1
			else:
				ret[i, j] = 0
	return ret

def laplaceOfGaussianOperator(padded_img, threshold):
	ret = np.zeros(padded_img.shape, np.int) 
	mask = np.array([[0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0],\
					[0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0],\
					[0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0],\
					[-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1],\
					[-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1],\
					[-2, -9, -23, -1, 103, 178, 103, -1, -23, -9, -2],\
					[-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1],\
					[-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1],\
					[0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0],\
					[0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0],\
					[0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0]])
	
	for i in range(padded_img.shape[0] - 10):
		for j in range(padded_img.shape[1] - 10):
			temp = 0
			for m in range(11):
				for n in range(11):
					temp += mask[m, n] * padded_img[i + m, j + n]
			# slide p.165
			if temp >= threshold:
				ret[i, j] = 1
			elif temp <= -threshold:
				ret[i, j] = -1
			else:
				ret[i, j] = 0
	return ret			
	
def differenceOfGaussianOperator(padded_img, threshold):
	ret = np.zeros(padded_img.shape, np.int) 
	mask = np.array([[-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1],\
						[-3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3],\
						[-4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4],\
						[-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6],\
						[-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7],\
						[-8, -13, -17, 15, 160, 283, 160, 15, -17, -13, -8],\
						[-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7],\
						[-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6],\
						[-4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4],\
						[-3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3],\
						[-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1]])
	
	for i in range(padded_img.shape[0] - 10):
		for j in range(padded_img.shape[1] - 10):
			temp = 0
			for m in range(11):
				for n in range(11):
					temp += mask[m, n] * padded_img[i + m, j + n]
			# slide p.165
			if temp >= threshold:
				ret[i, j] = 1
			elif temp <= -threshold:
				ret[i, j] = -1
			else:
				ret[i, j] = 0
	return ret			
	
def zeroCrossingOperator(padded_img, mask_size):
	center = int(mask_size // 2)
	ret = np.zeros(padded_img.shape, np.int) 
	for i in range(padded_img.shape[0]):
		for j in range(padded_img.shape[1]):
			_list = []
			for kernel_i in range(mask_size):
				for kernel_j in range(mask_size):
					scope_i = i + kernel_i - center
					scope_j = j + kernel_j - center
					if (0 <= scope_i and scope_i < length) and (0 <= scope_j and scope_j < width):
						_list.append(padded_img[scope_i, scope_j])
			if padded_img[i, j] == 1:
				if 1 in _list and -1 in _list: # this pixel has zero-crossing
					ret[i, j] = 0
				else:
					ret[i, j] = 255
			else:
				ret[i ,j] = 255
					
	real_ret = np.zeros((512, 512), np.int)
	for i in range(512):
		for j in range(512):
			real_ret[i, j] = ret[i + center, j + center]
	return real_ret

	
padded_img_for_mask_3 = padding(img, 2)
padded_img_for_mask_11 = padding(img, 10)
# laplace 1
laplace1_img = laplace1Operator(padded_img_for_mask_3, 15)
zeroCrossing_laplace1_img = zeroCrossingOperator(laplace1_img, 3)
cv2.imwrite('zeroCrossing_laplace1_img.png', zeroCrossing_laplace1_img)
# laplace 2
laplace2_img = laplace2Operator(padded_img_for_mask_3, 15)
zeroCrossing_laplace2_img = zeroCrossingOperator(laplace2_img, 3)
cv2.imwrite('zeroCrossing_laplace2_img.png', zeroCrossing_laplace2_img)
# Minimum variance Laplacian
minimumVarianceLaplacian_img = minimumVarianceLaplacianOperator(padded_img_for_mask_3, 20)
zeroCrossing_minimumVarianceLaplacian_img = zeroCrossingOperator(minimumVarianceLaplacian_img, 3)
cv2.imwrite('zeroCrossing_minimumVarianceLaplacian_img.png', zeroCrossing_minimumVarianceLaplacian_img)
# Laplace of Gaussian
laplaceOfGaussian_img = laplaceOfGaussianOperator(padded_img_for_mask_11, 3000)
zeroCrossing_laplaceOfGaussian_img = zeroCrossingOperator(laplaceOfGaussian_img, 3)
cv2.imwrite('zeroCrossing_laplaceOfGaussian_img.png', zeroCrossing_laplaceOfGaussian_img)
# Difference of Gaussian
differenceOfGaussian_img = differenceOfGaussianOperator(padded_img_for_mask_11, 1)
zeroCrossing_differenceOfGaussian_img = zeroCrossingOperator(differenceOfGaussian_img, 3)
cv2.imwrite('zeroCrossing_differenceOfGaussian_img.png', zeroCrossing_differenceOfGaussian_img)
