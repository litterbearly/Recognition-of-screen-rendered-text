# import the necessary packages
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2 as cv

image = cv.imread('4.jpg')
image = imutils.resize(image, height=500)
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
blurred = cv.GaussianBlur(gray, (5, 5), 0)
edged = cv.Canny(blurred, 25, 30, 255)
cv.imshow('edged', edged)

# find contours in the edge map, then sort them by their
# size in descending order
cnts = cv.findContours(edged.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv.contourArea, reverse=True)
displayCnt = None

# loop over the contours
for c in cnts:
	# approximate the contour
	peri = cv.arcLength(c, True)
	approx = cv.approxPolyDP(c, 0.02 * peri, True)

	# if the contour has four vertices, then we have found
	# the thermostat display
	if len(approx) == 4:
		displayCnt = approx
		break
# extract the thermostat display, apply a perspective transform
# to it
warped = four_point_transform(gray, displayCnt.reshape(4, 2))
output = four_point_transform(image, displayCnt.reshape(4, 2))
cv.imshow('output', output)
cv.waitKey(0)
cv.destroyAllWindows()