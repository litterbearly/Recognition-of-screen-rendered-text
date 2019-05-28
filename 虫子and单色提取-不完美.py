import cv2
import numpy as np


def get_image(image):
	# 提取某一颜色的画面
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	# 只读取绿色
	lower_hsv = np.array([35, 43, 46])
	upper_hsv = np.array([77, 255, 255])
	mask = cv2.inRange(hsv, lowerb=lower_hsv, upperb=upper_hsv)
	# mask 二值化，范围内为1，范围外为0
	# 只显示绿色内容
	dst = cv2.bitwise_and(image, image, mask=mask)
	gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

	return image, gray


def Gaussian_Blur(gray):
	# 高斯去噪
	blurred = cv2.GaussianBlur(gray, (5, 9), 0)

	return blurred


def Sobel_gradient(blurred):
	# 索比尔算子来计算x、y方向梯度
	gradX = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=1, dy=0)
	gradY = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=0, dy=1)

	gradient = cv2.subtract(gradX, gradY)
	gradient = cv2.convertScaleAbs(gradient)

	return gradX, gradY, gradient


def Thresh_and_blur(gradient):
	blurred = cv2.GaussianBlur(gradient, (9, 9), 0)
	(_, thresh) = cv2.threshold(blurred, 90, 255, cv2.cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	return thresh


def image_morphology(thresh):
	# 建立一个椭圆核函数
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
	# 执行图像形态学, 细节直接查文档，很简单
	closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
	closed = cv2.erode(closed, None, iterations=4)
	closed = cv2.dilate(closed, None, iterations=4)

	return closed


def findcnts_and_box_point(closed):
	# 这里opencv3返回的是三个参数
	(cnts, _) = cv2.findContours(closed.copy(),
	                                cv2.RETR_LIST,
	                                cv2.CHAIN_APPROX_SIMPLE)
	c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
	# compute the rotated bounding box of the largest contour
	rect = cv2.minAreaRect(c)
	box = np.int0(cv2.boxPoints(rect))

	return box


def drawcnts_and_cut(original_img, box):
	# 因为这个函数有极强的破坏性，所有需要在img.copy()上画
	# draw a bounding box arounded the detected barcode and display the image
	draw_img = cv2.drawContours(original_img.copy(), [box], -1, (0, 0, 255), 3)

	Xs = [i[0] for i in box]
	Ys = [i[1] for i in box]
	x1 = min(Xs)
	x2 = max(Xs)
	y1 = min(Ys)
	y2 = max(Ys)
	hight = y2 - y1
	width = x2 - x1
	crop_img = original_img[y1:y1 + hight, x1:x1 + width]

	return draw_img, crop_img, hight, width

def resize(image, n=3):
	return cv2.resize(image, (int(image.shape[1] / n), int(image.shape[0] / n)))

capture = cv2.VideoCapture('./实例/4.mp4')

while(capture.isOpened()):
	ret, frame = capture.read()
	if ret == True:
		original_img, gray = get_image(frame)
		blurred = Gaussian_Blur(gray)
		gradX, gradY, gradient = Sobel_gradient(blurred)
		thresh = Thresh_and_blur(gradient)
		closed = image_morphology(thresh)
		box = findcnts_and_box_point(closed)
		draw_img, crop_img, h, w = drawcnts_and_cut(original_img, box)

		# 把它们都显示出来看看
		cv2.imshow('original_img', resize(original_img))
		# cv2.imshow('blurred', blurred)
		# cv2.imshow('gradX', gradX)
		# cv2.imshow('gradY', gradY)
		# cv2.imshow('final', gradient)
		# cv2.imshow('thresh', resize(thresh))
		cv2.imshow('closed', resize(closed))
		cv2.imshow('draw_img', resize(draw_img))
		cv2.imshow('crop_img', resize(crop_img, 1/2))

		if cv2.waitKey(100) & 0xFF == 27:  # 27-ESC;  ord('q')按q跳出
			break
	else:
		break
capture.release()
cv2.destroyAllWindows()

