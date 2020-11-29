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

def RobertsOperator(padded_img, threshold):
	# padded size 514 * 514
	ret = np.zeros(padded_img.shape, np.int) 
	r1 = np.array([[-1, 0], [0, 1]])
	r2 = np.array([[0, -1], [1, 0]])
	for i in range(padded_img.shape[0] - 1):
		for j in range(padded_img.shape[1] - 1):
			temp1, temp2 = 0, 0
			for m in range(2):
				for n in range(2):
					temp1 += r1[m, n] * padded_img[i + m, j + n]
					temp2 += r2[m, n] * padded_img[i + m, j + n]
			gradient_magnitude = math.sqrt(temp1 ** 2 + temp2 ** 2)
			if gradient_magnitude >= threshold:
				ret[i, j] = 0
			else:
				ret[i, j] = 255
	
	# back to 512 * 512
	real_ret = np.zeros((512, 512), np.int)
	for i in range(512):
		for j in range(512):
			real_ret[i, j] = ret[i + 1, j + 1]
	return real_ret
	
def pretwittOperator(padded_img, threshold):
	# padded size 514 * 514
	ret = np.zeros(padded_img.shape, np.int) 
	r1 = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
	r2 = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
	for i in range(padded_img.shape[0] - 2):
		for j in range(padded_img.shape[1] - 2):
			temp1, temp2 = 0, 0
			for m in range(3):
				for n in range(3):
					temp1 += r1[m, n] * padded_img[i + m, j + n]
					temp2 += r2[m, n] * padded_img[i + m, j + n]
			gradient_magnitude = math.sqrt(temp1 ** 2 + temp2 ** 2)
			if gradient_magnitude >= threshold:
				ret[i, j] = 0
			else:
				ret[i, j] = 255
	
	# back to 512 * 512
	real_ret = np.zeros((512, 512), np.int)
	for i in range(512):
		for j in range(512):
			real_ret[i, j] = ret[i + 1, j + 1]
	return real_ret			
			
def sobelOperator(padded_img, threshold):
	# padded size 514 * 514
	ret = np.zeros(padded_img.shape, np.int) 
	r1 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
	r2 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
	for i in range(padded_img.shape[0] - 2):
		for j in range(padded_img.shape[1] - 2):
			temp1, temp2 = 0, 0
			for m in range(3):
				for n in range(3):
					temp1 += r1[m, n] * padded_img[i + m, j + n]
					temp2 += r2[m, n] * padded_img[i + m, j + n]
			gradient_magnitude = math.sqrt(temp1 ** 2 + temp2 ** 2)
			if gradient_magnitude >= threshold:
				ret[i, j] = 0
			else:
				ret[i, j] = 255
	
	# back to 512 * 512
	real_ret = np.zeros((512, 512), np.int)
	for i in range(512):
		for j in range(512):
			real_ret[i, j] = ret[i + 1, j + 1]
	return real_ret		

def freiAndChenOperator(padded_img, threshold):
	# padded size 514 * 514
	ret = np.zeros(padded_img.shape, np.int) 
	r1 = np.array([[-1, -math.sqrt(2), -1], [0, 0, 0], [1, math.sqrt(2), 1]])
	r2 = np.array([[-1, 0, 1], [-math.sqrt(2), 0, math.sqrt(2)], [-1, 0, 1]])
	for i in range(padded_img.shape[0] - 2):
		for j in range(padded_img.shape[1] - 2):
			temp1, temp2 = 0, 0
			for m in range(3):
				for n in range(3):
					temp1 += r1[m, n] * padded_img[i + m, j + n]
					temp2 += r2[m, n] * padded_img[i + m, j + n]
			gradient_magnitude = math.sqrt(temp1 ** 2 + temp2 ** 2)
			if gradient_magnitude >= threshold:
				ret[i, j] = 0
			else:
				ret[i, j] = 255
	
	# back to 512 * 512
	real_ret = np.zeros((512, 512), np.int)
	for i in range(512):
		for j in range(512):
			real_ret[i, j] = ret[i + 1, j + 1]
	return real_ret			
	
def kirschCompassOperator(padded_img, threshold):
	# padded size 514 * 514
	ret = np.zeros(padded_img.shape, np.int) 
	r1 = np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]])
	r2 = np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]])
	r3 = np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]])
	r4 = np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]])
	r5 = np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]])
	r6 = np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]])
	r7 = np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]])
	r8 = np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]])
	
	for i in range(padded_img.shape[0] - 2):
		for j in range(padded_img.shape[1] - 2):
			temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8 = 0, 0, 0, 0, 0, 0, 0, 0
			for m in range(3):
				for n in range(3):
					temp1 += r1[m, n] * padded_img[i + m, j + n]
					temp2 += r2[m, n] * padded_img[i + m, j + n]
					temp3 += r3[m, n] * padded_img[i + m, j + n]
					temp4 += r4[m, n] * padded_img[i + m, j + n]
					temp5 += r5[m, n] * padded_img[i + m, j + n]
					temp6 += r6[m, n] * padded_img[i + m, j + n]
					temp7 += r7[m, n] * padded_img[i + m, j + n]
					temp8 += r8[m, n] * padded_img[i + m, j + n]
					
			compare_array = [temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8]
			
			if max(compare_array) >= threshold:
				ret[i, j] = 0
			else:
				ret[i, j] = 255
	
	# back to 512 * 512
	real_ret = np.zeros((512, 512), np.int)
	for i in range(512):
		for j in range(512):
			real_ret[i, j] = ret[i + 1, j + 1]
	return real_ret				
	
def robinsonCompassOperator(padded_img, threshold):
	# padded size 514 * 514
	ret = np.zeros(padded_img.shape, np.int) 
	r1 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, -0, 1]])
	r2 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]])
	r3 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
	r4 = np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]])
	r5 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
	r6 = np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]])
	r7 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
	r8 = np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]])
	
	for i in range(padded_img.shape[0] - 2):
		for j in range(padded_img.shape[1] - 2):
			temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8 = 0, 0, 0, 0, 0, 0, 0, 0
			for m in range(3):
				for n in range(3):
					temp1 += r1[m, n] * padded_img[i + m, j + n]
					temp2 += r2[m, n] * padded_img[i + m, j + n]
					temp3 += r3[m, n] * padded_img[i + m, j + n]
					temp4 += r4[m, n] * padded_img[i + m, j + n]
					temp5 += r5[m, n] * padded_img[i + m, j + n]
					temp6 += r6[m, n] * padded_img[i + m, j + n]
					temp7 += r7[m, n] * padded_img[i + m, j + n]
					temp8 += r8[m, n] * padded_img[i + m, j + n]
					
			compare_array = [temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8]
			
			if max(compare_array) >= threshold:
				ret[i, j] = 0
			else:
				ret[i, j] = 255
	
	# back to 512 * 512
	real_ret = np.zeros((512, 512), np.int)
	for i in range(512):
		for j in range(512):
			real_ret[i, j] = ret[i + 1, j + 1]
	return real_ret				
	
def nevatiaBabuOperator(padded_img, threshold):
	# padded size 516 * 516
	ret = np.zeros(padded_img.shape, np.int) 
	r1 = np.array([[100, 100, 100, 100, 100], [100, 100, 100, 100, 100], [0, 0, 0, 0, 0], [-100, -100, -100, -100, -100], [-100, -100, -100, -100, -100]])
	r2 = np.array([[100, 100, 100, 100, 100], [100, 100, 100, 78, -32], [100, 92, 0, -92, -100], [32, -78, -100, -100, -100], [-100, -100, -100, -100, -100]])
	r3 = np.array([[100, 100, 100, 32, -100], [100, 100, 92, -78, -100], [100, 100, 0, -100, -100], [100, 78, -92, -100, -100], [100, -32, -100, -100, -100]])
	r4 = np.array([[-100, -100, 0, 100, 100], [-100, -100, 0, 100, 100], [-100, -100, 0, 100, 100], [-100, -100, 0, 100, 100], [-100, -100, 0, 100, 100]])
	r5 = np.array([[-100, 32, 100, 100, 100], [-100, -78, 92, 100, 100], [-100, -100, 0, 100, 100], [-100, -100, -92, 78, 100], [-100, -100, -100, -32, 100]])
	r6 = np.array([[100, 100, 100, 100, 100], [-32, 78, 100, 100, 100], [-100, -92, 0, 92, 100], [-100, -100, -100, -78, 32], [-100, -100, -100, -100, -100]])	
	for i in range(padded_img.shape[0] - 4):
		for j in range(padded_img.shape[1] - 4):
			temp1, temp2, temp3, temp4, temp5, temp6 = 0, 0, 0, 0, 0, 0
			for m in range(5):
				for n in range(5):
					temp1 += r1[m, n] * padded_img[i + m, j + n]
					temp2 += r2[m, n] * padded_img[i + m, j + n]
					temp3 += r3[m, n] * padded_img[i + m, j + n]
					temp4 += r4[m, n] * padded_img[i + m, j + n]
					temp5 += r5[m, n] * padded_img[i + m, j + n]
					temp6 += r6[m, n] * padded_img[i + m, j + n]
					
			compare_array = [temp1, temp2, temp3, temp4, temp5, temp6]
			
			if max(compare_array) >= threshold:
				ret[i, j] = 0
			else:
				ret[i, j] = 255
	
	# back to 512 * 512
	real_ret = np.zeros((512, 512), np.int)
	for i in range(512):
		for j in range(512):
			real_ret[i, j] = ret[i + 2, j + 2]
	return real_ret					
	
padded_img_for_mask_3 = padding(img, 2)
padded_img_for_mask_5 = padding(img, 4)

robert_img = RobertsOperator(padded_img_for_mask_3, 12)
cv2.imwrite('robert_img.png', robert_img)

pretwitt_img = pretwittOperator(padded_img_for_mask_3, 24)
cv2.imwrite('pretwitt_img.png', pretwitt_img)

sobel_img = sobelOperator(padded_img_for_mask_3, 38)
cv2.imwrite('sobel_img.png', sobel_img)

freiAndChen_img = freiAndChenOperator(padded_img_for_mask_3, 30)
cv2.imwrite('freiAndChen_img.png', freiAndChen_img)

kirschCompass_img = kirschCompassOperator(padded_img_for_mask_3, 135)
cv2.imwrite('kirschCompass_img.png', kirschCompass_img)

robinsonCompass_img = robinsonCompassOperator(padded_img_for_mask_3, 43)
cv2.imwrite('robinsonCompass_img.png', robinsonCompass_img)

nevatiaBabu_img = nevatiaBabuOperator(padded_img_for_mask_5, 12500)
cv2.imwrite('nevatiaBabu_img.png', nevatiaBabu_img)