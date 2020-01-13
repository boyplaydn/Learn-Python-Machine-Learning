import numpy as np
import cv2
from sklearn.externals import joblib


model = joblib.load("train_number.py")

# Load an color image in grayscale
img = cv2.imread('img.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img_gray = cv2.GaussianBlur(img_gray, (5,5), 0)
ret, im_th = cv2.threshold(img_gray, 115, 255, cv2.THRESH_BINARY_INV)

# canny = cv2.Canny(im_th, 70, 170)
# cv2.imshow('canny', canny)

ctrs,_ = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

rects = [cv2.boundingRect(ctr) for ctr in ctrs]
font = cv2.FONT_HERSHEY_COMPLEX

for rect in rects:
    print(rect[0],'-',rect[1])
    print(rect[0]+rect[2] ,'-', rect[1] + rect[3])
    leng = int(rect[3] * 1.6)
    pt1 = int(rect[1] + rect[3]//2 - leng//2)
    pt2 = int(rect[0] + rect[2]//2 - leng//2)
    roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
    roi = cv2.resize(roi, (28,28), interpolation=cv2.INTER_AREA) #bien roi thanh mang 2 chieu voi kich thuoc 28*28
    roi = cv2.dilate(roi, (3,3))
    number = np.array([roi]).reshape(1, 28*28) #bien roi thanh mang 1 chieu voi kich thuoc 28*28
    predict = model.predict(number)
    print('prediction', str(int(predict[0])))
    print(roi)
    cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 0), 3)
    cv2.rectangle(im_th, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 0), 3)
    cv2.putText(img, str(int(predict[0])),(rect[0]-10,rect[1]-10), font, 1,(0,0,0),2,cv2.LINE_AA)
cv2.imshow('image',img)

#cv2.imshow('img_threshold', im_th)
cv2.waitKey(0)
cv2.destroyAllWindows()