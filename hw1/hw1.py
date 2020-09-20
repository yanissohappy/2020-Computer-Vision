import cv2

img = cv2.imread('lena.bmp')
img_binarized = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
length = len(img) # row
width = len(img[0]) # col, it's actually a square though

def upsideDown(image):
	copy1 = image.copy()
	if length & 1 == 0: # if even rows
		for i in range(int(length // 2)):
			copy1[i], copy1[length - i - 1] = image[length - i - 1], image[i]	
	else: # if odd rows
		for i in range(int(length // 2)):
			if i == int(length // 2): 
				copy1[i] = image[i]
			else: 
				copy1[i], copy1[length - i - 1] = image[length - i - 1], image[i]
	return copy1
	
def rightSideLeft(image):
	copy2 = image.copy()
	if width & 1 == 0: # if even rows
		for i in range(int(width // 2)):
			for j in range(int(length)):
				copy2[j][i], copy2[j][width - i - 1] = image[j][width - i - 1], image[j][i]	
	else: # if odd rows
		for i in range(int(width // 2)):
			for j in range(int(length)):
				if i == int(width // 2): 
					copy2[j][i] = image[j][i]
				else: 
					copy2[j][i], copy2[j][width - i - 1] = image[j][width - i - 1], image[j][i]
	return copy2	
	
def diagonallyFlip(image):
	copy3 = image.copy()
	for j in range(int(length)):
		for i in range(int(j) + 1):
			if i == j:
				copy3[j][i] = image[j][i]
			else:
				copy3[j][i], copy3[i][j] = image[i][j], image[j][i]
	return copy3
	
def binarize(image):
	copy4 = image.copy()
	for i in range(int(width)):
		for j in range(int(length)):
			if image[j][i] >= 128: # 128 ~ 255
				copy4[j][i] = 255
			else: # 0 ~ 127
				copy4[j][i] = 0
	return copy4

copy1 = upsideDown(img)
copy2 = rightSideLeft(img)
copy3 = diagonallyFlip(img)
copy4 = binarize(img_binarized)
cv2.imwrite('upside_down.jpg', copy1)
cv2.imwrite('right_side_left.jpg', copy2)
cv2.imwrite('diagonally_flip.jpg', copy3)
cv2.imwrite('binary.jpg', copy4)
