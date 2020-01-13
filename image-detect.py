import numpy as np
import cv2

# Load an color image in grayscale
img = cv2.imread('stand.png')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_red = np.array([150, 150, 0])
upper_red = np.array([255, 255, 255])

mask = cv2.inRange(hsv, lower_red, upper_red)

ctrs,_ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

rects = [cv2.boundingRect(ctr) for ctr in ctrs]


for rect in rects:
    print(rect[0],'-',rect[1])
    print(rect[0]+rect[2] ,'-', rect[1] + rect[3])
    cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 0), 3)


cv2.imshow('image',img)
cv2.imshow('mask',mask)
cv2.waitKey(0)
cv2.destroyAllWindows()