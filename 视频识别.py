import cv2 as cv
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import numpy as np
import matplotlib.pyplot as plt


fourcc = cv.VideoWriter_fourcc(*'XVID')  # 设置视频编码格式，输出为avi
out = cv.VideoWriter('output.avi', fourcc, 20.0, (1920, 1080))  # 名称， 格式， 帧率， 帧大小


# 播放视频
def extraxt_object_demo():
	capture = cv.VideoCapture('测试1.mp4')
	while(True):
		ret, frame = capture.read()
		if ret == True:
			image = imutils.resize(frame, height=500)
			gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
			blurred = cv.GaussianBlur(gray, (5, 5), 0)
			edged = cv.Canny(blurred, 25, 30, 255)
			cv.imshow('edged', edged)
			c = cv.waitKey(50)  # 等待50ms扫描一次
			if c == 27:  # 按ESC停止摄像头输入（不会关闭窗口）
				break
		else:
			break

	capture.release()
	cv.destroyAllWindows()


extraxt_object_demo()
