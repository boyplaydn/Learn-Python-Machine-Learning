import cv2
import numpy as np
from sklearn.externals import joblib

cap = cv2.VideoCapture(0)
model = joblib.load("train_number.py")
while True:
    _, img = cap.read()
    flip_img = cv2.flip(img, 1)
    img_gray = cv2.cvtColor(flip_img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (5,5), 0)
    ret, im_th = cv2.threshold(img_gray, 115, 255, cv2.THRESH_BINARY_INV)

    # canny = cv2.Canny(im_th, 70, 170)
    # cv2.imshow('canny', canny)

    ctrs,_ = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = sorted([cv2.boundingRect(ctr) for ctr in ctrs], key = lambda x: x[1])
    font = cv2.FONT_HERSHEY_COMPLEX

    for rect in rects:
        if(rect[3]  + rect[0] >= 60) and (rect[3] + rect[2]) >= 60:
            leng = int(rect[3] * 1.6)
            pt1 = int(rect[1] + rect[3]//2 - leng//2)
            pt2 = int(rect[0] + rect[2]//2 - leng//2)
            try:
                roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
                roi = cv2.resize(roi, (28,28), interpolation=cv2.INTER_AREA) #bien roi thanh mang 2 chieu voi kich thuoc 28*28
                roi = cv2.dilate(roi, (3,3))
                number = np.array([roi]).reshape(1, 28*28) #bien roi thanh mang 1 chieu voi kich thuoc 28*28
                predict = model.predict(number)
                cv2.rectangle(flip_img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 0), 3)
                cv2.putText(flip_img, str(int(predict[0])),(rect[0]-10,rect[1]-10), font, 1,(0,0,0),2,cv2.LINE_AA)
                print('prediction', str(int(predict[0])))
            except:
                pass
            
            
        #cv2.rectangle(flip_img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 0), 3)
      

    cv2.imshow('image',flip_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()