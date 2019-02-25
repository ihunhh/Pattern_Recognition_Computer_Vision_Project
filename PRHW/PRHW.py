
import numpy as np
from PIL import Image as im
#import matplotlib.pyplot as plt
#from skimage import io 
from sklearn import preprocessing

IMG_PATH_Q1 = 'swan.png'
IMG_PATH_Q2 = 'face.jpg'
IMG_PATH_Q3 = 'cat.png'
IMG_PATH_Q4_cat = 'cat.png'
IMG_PATH_Q4_face = 'face.jpg'
IMG_PATH_Q4_face_msk = 'faceMask.png'

FILTER_Q1 = np.array([[-1, -3, -1], [0, 0, 0], [1, 3 ,1]])
FILTER_Q2 = np.array([[1, 2, 0, -2, -1]])
FILTER_Q4 = np.array([[-1, 0, 1]])
WEIGHT_Q3 = np.array([[-3, 1, -3]])
WEIGHT_Q4 = np.array([[1, -2, 1]])
MATRIX_Q4_R = np.array([[11, 12, 13], [14, 15, 16], [17, 18, 19]]) 
MATRIX_Q4_G = np.array([[21, 22, 23], [24, 25, 26], [27, 28, 29]]) 
MATRIX_Q4_B = np.array([[31, 32, 33], [34, 35, 36], [37, 38, 39]]) 
MATRIX_Q4_M = np.array([[1, 0, -1], [0, 0, 1], [-1, 1, 1]]) 


def imgread_to_data(path):
	img = im.open(path)
#	plt.imshow(img)
#	io.show()
	width, height = img.size
	data = np.array(img)
	
#	print(img.size)
#	print(data.shape)	
#	print(np.shape(data))
#	plt.imshow(data)
#	io.show()

	return data


def zero_padding(data, psize_h, psize_w):
#	print(data.size)
	image_pad = np.pad(data, ((int(psize_h), int(psize_h)), (int(psize_w), int(psize_w))), 'constant')
#	img = im.fromarray(image_pad)
#	img.save("your_file.jpeg")
#	plt.imshow(img)
#	io.show()
	
	return image_pad

def applyFilter(data, ft):
	temp = []
	p_sum = 0
	
	f_shape = list(np.shape(ft))
	fts_w = int(f_shape[1])
	fts_h = int(f_shape[0])
#	max_f_shape = max(f_shape)
#	print(max_f_shape)
#	data = imgread_to_data(img_in)		
	img_size = list(np.shape(data))	

	data_pad = zero_padding(data, (fts_h - 1) / 2, (fts_w - 1) / 2)
	d_shape = list(np.shape(data_pad))
#	print(d_shape)
#	print(f_shape)
	for y in range(int(d_shape[0]) - int(f_shape[0]) + 1):		
		for x in range(int(d_shape[1]) - int(f_shape[1]) + 1):
			for h in range(int(f_shape[0])):
				for w in range(int(f_shape[1])):
					p_sum = p_sum + (data_pad[y + h][x + w] * ft[h][w])
			temp.append(p_sum)
			p_sum = 0		
	new_data = np.asarray(temp)
#	print('ads_sum = ' + str(np.sum(np.absolute(new_data))))
	new_data = np.asarray(temp, dtype=np.float64)
#	print(new_data)
	new_data = preprocessing.minmax_scale(new_data, feature_range=(0, 255)).astype(np.uint8)
#	print(new_data)
	new_data = new_data.reshape((int(img_size[0]), int(img_size[1])))
#	print(new_data)
	
#	plt.imshow(img)
#	io.show()	
	return new_data

def grayscale(data):
	d_shape = list(np.shape(data))
#	print(list(np.shape(data)))
	t_data = np.transpose(data, (2, 0, 1))
	t_data_p = np.reshape(t_data, (3, int(d_shape[0]) * int(d_shape[1])))
	data_gray_f = np.sum(t_data_p, axis=0) / 3
	data_gray = np.reshape(data_gray_f, (d_shape[0], d_shape[1])).astype(np.uint8)
#	img = im.fromarray(t_data_gray)
#	img.save("faceGray.png")	

#	print(list(np.shape(t_data)))
#	print(t_data_gray)
	return data_gray

def computeEngGrad(data_gray, ft):
	dg_shape = list(np.shape(data_gray))
	part_x = applyFilter(data_gray, ft)
	img_x = im.fromarray(part_x)
	img_x.save("x.jpg")
#	plt.imshow(img_1)
#	io.show()
	part_y = applyFilter(data_gray, np.transpose(ft, (1, 0)))
	img_y = im.fromarray(part_y)
	img_y.save("y.jpg")
#	plt.imshow(img_2)
#	io.show()
	part_x = part_x.astype(np.float64)
	part_y = part_y.astype(np.float64)
	_sum = np.sqrt(np.sum([np.power(np.absolute(part_x.flatten()), 2), np.power(np.absolute(part_y.flatten()), 2)], axis=0))
#	print('Q2_ads_sum = ' + str(np.sum(_sum)))

#	print(_sum)
#	part_1_img = im.fromarray(np.reshape(preprocessing.minmax_scale(np.power(np.absolute(part_2.flatten()), 2), feature_range=(0, 255)).astype(np.uint8) , (int(dg_shape[0]), int(dg_shape[1]))))
#	part_1_img.save('part1.jpg')
	_sum = preprocessing.minmax_scale(_sum, feature_range=(0, 255)).astype(np.uint8)
#	print(_sum)
	_sum = np.reshape(_sum , (int(dg_shape[0]), int(dg_shape[1])))
#	img_3 = im.fromarray(_sum)
#	plt.imshow(img_3)
#	io.show()
#	img_3.save("faceEngG.jpg")
	return _sum, part_x.astype(np.uint8), part_y.astype(np.uint8)

def computeEngColor(data, W):
#	data_int32 = data.astype(np.int32)
	d_shape = list(np.shape(data))
	data_t = data.transpose(2, 0, 1)
#	print(np.shape(data_int32))
	data_reshp = data_t.reshape((3, int(d_shape[0]) * int(d_shape[1])))

	data_out_wo_norm = np.multiply(data_reshp[0], W[0][0]) + np.multiply(data_reshp[1], W[0][1]) + np.multiply(data_reshp[2], W[0][2])
	data_out_wo_norm = data_out_wo_norm.reshape((int(d_shape[0]), int(d_shape[1])))

#	data_output = preprocessing.minmax_scale(data_out, feature_range=(0, 255)).astype(np.float64).reshape((int(d_shape[0]), int(d_shape[1])))
#	print(np.shape(data_int32))
#	print(np.shape(data_int32_reshp))
#	print(np.shape(data_out))
#	print(data_int32_reshp)
#	print(data_out)
#	print(np.shape(data_output))
#	print(data_output)
	return data_out_wo_norm

def computeEng(data, F, W, maskW):
	data = data.transpose(2, 0, 1)
	d_shape_h = int(list(np.shape(data))[1])
	d_shape_w = int(list(np.shape(data))[2])
	data_color = np.array(data[0:3, :, :]).transpose(1, 2, 0)
	data_msk = np.array(data[3, :, :])
#	print(data_color)
	data_gray = grayscale(data_color)
	cEG, _t1, _t2 = computeEngGrad(data_gray, F)

	mskW_x_m = maskW * data_msk 
#	print(mskW_x_m)

	_sum_eng = np.sum([cEG.flatten(), computeEngColor(data_color, W).flatten(), mskW_x_m.flatten()], axis=0).reshape((d_shape_h, d_shape_w))

	return _sum_eng

def removeSeamV(data_im4, seam):
	data_im4_t_data = data_im4.transpose(2, 0, 1)
	d_im4_shape = list(np.shape(data_im4_t_data))
	c_shape = d_im4_shape[0]
	s_shape = list(np.shape(seam))
	data_new = np.zeros((d_im4_shape[0], d_im4_shape[1], d_im4_shape[2] - 1))
#	print('data_new:')
#	print(np.shape(data_new))
	if not d_im4_shape[1] == s_shape[0]:
		raise AssertionError('size not match!!')
	
	for x in range(c_shape):
		for y in range(d_im4_shape[1]):
#			print(np.shape(np.delete(data_im4_t_data[x][y], seam[y])))
			data_new[x][y] = np.delete(data_im4_t_data[x][y], seam[y])
	data_new = data_new.transpose(1, 2, 0)
	return data_new

def addSeamV(data_im4, seam):
	data_im4_t_data = data_im4.transpose(2, 0, 1)
	d_im4_shape = list(np.shape(data_im4_t_data))
	c_shape = d_im4_shape[0]
	s_shape = list(np.shape(seam))
	data_new = np.zeros((d_im4_shape[0], d_im4_shape[1], d_im4_shape[2] + 1))
#	print(shape(data_new))
#	print('data:')
#	print(np.shape(data_im4_t_data))
#	print('data_seam:')
#	print(np.shape(seam))
	if not d_im4_shape[1] == s_shape[0]:
		raise AssertionError('size not match!!')
	
	for x in range(c_shape):
		for y in range(d_im4_shape[1]): 
			
			data_new[x][y] = np.insert(data_im4_t_data[x][y], seam[y], data_im4_t_data[x][y][seam[y]])
	data_new = data_new.transpose(1, 2, 0)		
	return data_new

def seamV_DP(data_E):
	d_shape = list(np.shape(data_E))
	M = np.zeros(np.shape(data_E))
	P = np.zeros(np.shape(data_E), dtype=np.uint32)
#	print(data_E)
	for x in range(d_shape[0]):
		if x == 0:
			M[x] = data_E[x]
			P[x] = np.full_like(P[x], np.nan)
#			print(P)
		else:
			for y in range(d_shape[1]):
				if y == 0:
					if M[x - 1][y] <= M[x - 1][y + 1]:
						M[x][y] = M[x - 1][y] + data_E[x][y]
						P[x][y] = y
					else:
						M[x][y] = M[x - 1][y + 1] + data_E[x][y]
						P[x][y] = y + 1	
				elif y == (d_shape[1] - 1):
					if M[x - 1][y] <= M[x - 1][y - 1]:
						M[x][y] = M[x - 1][y] + data_E[x][y]
						P[x][y] = y
					else:
						M[x][y] = M[x - 1][y - 1] + data_E[x][y]
						P[x][y] = y - 1	
				else:					
					if (M[x - 1][y - 1] <= M[x - 1][y]) and (M[x - 1][y - 1] <= M[x - 1][y + 1]):
						M[x][y] = M[x - 1][y - 1] + data_E[x][y]
						P[x][y] = y - 1	
					elif (M[x - 1][y] <= M[x - 1][y - 1]) and (M[x - 1][y] <= M[x - 1][y + 1]):
						M[x][y] = M[x - 1][y] + data_E[x][y]
						P[x][y] = y	
					else:
						M[x][y] = M[x - 1][y + 1] + data_E[x][y]
						P[x][y] = y + 1
#	print(P)
#	print(M)
	return M, P

def bestSeamV(M, P):
	M_shape = list(np.shape(M))
	seam = np.zeros(M_shape[0], dtype=np.uint32)
	for x in range(M_shape[1]):
		if x == 0:
			min_ = M[M_shape[0] - 1][x]
			min_loc = x
		else:
			if min_ >= M[M_shape[0] - 1][x]:
				min_ = M[M_shape[0] - 1][x]
				min_loc = x
	c = min_
	for x in range(M_shape[0]):
		if x == 0:
			seam[M_shape[0] - 1 - x] = min_loc
			temp_loc = P[M_shape[0] - 1 - x][min_loc]
#		elif x == (M_shape[0] - 1):
#			seam[M_shape[0] - 1 - x] = P[M_shape[0] - 1 - x][temp_loc]

		else:
			seam[M_shape[0] - 1 - x] = P[M_shape[0] - 1 - x + 1][temp_loc]
			temp_loc = P[M_shape[0] - 1 - x + 1][temp_loc]
#			print(M_shape[0])
#	print(seam)		
	return seam, c


def reduceWidth(data_im4, data_E):
	M, P = seamV_DP(data_E)
	seam, c = bestSeamV(M, P)
	im4Out = removeSeamV(data_im4, seam)

	return seam, im4Out, c

def reduceHeight(data_im4, data_E):
	data_im4_t = data_im4.transpose(1, 0, 2)
	data_E_t = data_E.transpose(1, 0)
	M, P = seamV_DP(data_E_t)
	seam, c = bestSeamV(M, P)
	im4Out = removeSeamV(data_im4_t, seam).transpose(1, 0, 2)


	return seam, im4Out, c

def increaseWidth(data_im4, data_E):
	M, P = seamV_DP(data_E)
	seam, c = bestSeamV(M, P)
	im4Out = addSeamV(data_im4, seam)

	return seam, im4Out, c

def increaseHeight(data_im4, data_E):
	data_im4_t = data_im4.transpose(1, 0, 2)
	data_E_t = data_E.transpose(1, 0)
	M, P = seamV_DP(data_E_t)
	seam, c = bestSeamV(M, P)
	im4Out = addSeamV(data_im4_t, seam).transpose(1, 0, 2)

	return seam, im4Out, c

def intelligentResize(data_im , v, h, W, mask, maskWeight):
	data_im_shape = list(np.shape(data_im))
	data_msk_shape = list(np.shape(mask))
	if not (data_im_shape[0] == data_msk_shape[0]) and (data_im_shape[1] == data_msk_shape[1]):
		raise AssertionError('size of msk is not match with img!!')
	data_im_t = data_im.transpose(2, 0, 1)
	data_im4_t = np.append(data_im_t, [mask], axis=0)
	data_im4_data = data_im4_t.transpose(1, 2, 0)
#	gray_img = grayscale(data_im)
#	eng = computeEng(data_im4_data, FILTER_Q4, W, maskWeight)

	c = 0
	if v < 0:
		for x in range(abs(v)):
			eng = computeEng(data_im4_data, FILTER_Q4, W, maskWeight)
			_vs, data_im4_data, c_temp = reduceWidth(data_im4_data, eng)

			c = c + c_temp
	elif v > 0:
		for x in range(v):

			eng = computeEng(data_im4_data, FILTER_Q4, W, maskWeight)
			_vs, data_im4_data, c_temp = increaseWidth(data_im4_data, eng)
			c = c + c_temp

	if h < 0:
		for x in range(abs(h)):
			eng = computeEng(data_im4_data, FILTER_Q4, W, maskWeight)
			_hs, data_im4_data, c_temp = reduceHeight(data_im4_data, eng)
			c = c + c_temp
	elif h > 0:
		for x in range(h):
			print(x)
			eng = computeEng(data_im4_data, FILTER_Q4, W, maskWeight)
			_hs, data_im4_data, c_temp = increaseHeight(data_im4_data, eng)	
			c = c + c_temp
	totalCost = c	
	imOut = np.delete(data_im4_data.transpose(2, 0, 1), 3, axis=0).transpose(1, 2, 0)

	return totalCost, imOut

if __name__ == '__main__':

	data_Q1 = imgread_to_data(IMG_PATH_Q1)
	img_Q1_data = applyFilter(data_Q1, FILTER_Q1)
	print('sum of swanFiltered:' + str(np.sum(np.abs(img_Q1_data.flatten()))))
	img_Q1 = im.fromarray(img_Q1_data)
	img_Q1.save("swanFiltered.png")



	data_gray_Q2 = grayscale(imgread_to_data(IMG_PATH_Q2))
#	print(list(np.shape(data_Q2)))
	img_Q2_gray = im.fromarray(data_gray_Q2)
	img_Q2_gray.save("faceGray.png")	
	data_Q2, _x, _y = computeEngGrad(data_gray_Q2, FILTER_Q2)
	print('sum of faceEngG:' + str(np.sum(data_Q2.flatten())))
	_x = im.fromarray(_x)
	_x.save("faceX.jpg")
	_y = im.fromarray(_y)
	_y.save("faceY.jpg")
	img_Q2 = im.fromarray(data_Q2)
	img_Q2.save("faceEngG.jpg")	



	data_Q3 = imgread_to_data(IMG_PATH_Q3)
	data_Q3_out = computeEngColor(data_Q3, WEIGHT_Q3)
	d_shape_Q3 = list(np.shape(data_Q3_out))
#	print(d_shape_Q3)
	data_Q3_out = data_Q3_out.flatten().astype(np.float64)
	print('sum of catEngC:' + str(np.sum(data_Q3_out)))
	data_Q3_out = preprocessing.minmax_scale(data_Q3_out, feature_range=(0, 255)).reshape((d_shape_Q3[0], d_shape_Q3[1]))
	data_Q3_out_img = data_Q3_out.astype(np.uint8)
	img_output = im.fromarray(data_Q3_out_img)
	img_output.save("catEngC.png")


	data_im4_Q4 = np.array([MATRIX_Q4_R, MATRIX_Q4_G, MATRIX_Q4_B, MATRIX_Q4_M])
	data_im4_Q4_t = data_im4_Q4.transpose(1, 2, 0)
	data_im4_Q4_t_d = data_im4_Q4_t.astype(np.uint8)
	img_im4_Q4 = im.fromarray(data_im4_Q4_t_d)
	img_im4_Q4.save("im4.png")	
	computeEng(data_im4_Q4_t, FILTER_Q2, WEIGHT_Q3, 10)
#	print(np.shape(data_im4_Q4_t))
#	print(data_im4_Q4_t)
	data_Q4_cat = imgread_to_data(IMG_PATH_Q4_cat)
	catCost, data_Q4_cat_Out = intelligentResize(data_Q4_cat, -20, -20, WEIGHT_Q4, np.zeros((np.shape(data_Q4_cat)[0], np.shape(data_Q4_cat)[1])), 0)
	data_Q4_cat_Out_f = data_Q4_cat_Out.flatten().astype(np.float64)
	data_Q4_cat_Out = preprocessing.minmax_scale(data_Q4_cat_Out_f, feature_range=(0, 255)).reshape(np.shape(data_Q4_cat_Out)).astype(np.uint8)
	img_im4_Q4_cat = im.fromarray(data_Q4_cat_Out)
	img_im4_Q4_cat.save("catResized.png")
	print('the cat of the total cost of all seams:' + str(catCost))
	data_Q4_face = imgread_to_data(IMG_PATH_Q4_face)
	data_Q4_face_msk = imgread_to_data(IMG_PATH_Q4_face_msk)
	faceCost, data_Q4_face_Out = intelligentResize(data_Q4_face, -20, -20, WEIGHT_Q4, data_Q4_face_msk, -100)
	data_Q4_face_Out_f = data_Q4_face_Out.flatten().astype(np.float64)
	data_Q4_face_Out = preprocessing.minmax_scale(data_Q4_face_Out_f, feature_range=(0, 255)).reshape(np.shape(data_Q4_face_Out)).astype(np.uint8)
	img_im4_Q4_face = im.fromarray(data_Q4_face_Out)
	img_im4_Q4_face.save("faceResized.png")
	print('the face of the total cost of all seams:' + str(faceCost))



