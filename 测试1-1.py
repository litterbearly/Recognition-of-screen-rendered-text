import cv2 as cv
capture = cv.VideoCapture('3-1.mp4')
fourcc = cv.VideoWriter_fourcc(*'XVID')  # 设置视频编码格式，输出为avi
out = cv.VideoWriter('output1.avi', fourcc, 20.0, (int(capture.get(3)), int(capture.get(4))), 0)  # 名称， 格式， 帧率， 帧大小
while(capture.isOpened()):
	ret, frame = capture.read()
	if ret == True:
		cv.imshow('frame', frame)
		gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
		blurred = cv.GaussianBlur(gray, (5, 5), 0)
		edged = cv.Canny(blurred, 25, 30, 255)
		cv.imshow('edged', edged)
		out.write(edged)
		if cv.waitKey(1) & 0xFF == 27:  # 27-ESC;  ord('q')按q跳出
			break
	else:
		break
capture.release()
cv.destroyAllWindows()

